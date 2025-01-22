import os

import torch
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import clusterability, get_device


class Trainer:
    def __init__(self, model, tokenizer, batch_size, num_clusters):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = self.model.config.num_hidden_layers
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.device = get_device()
        self.max_grad_norm = 1.0
        self.val_freq = 100
        self.best_val_loss = float("inf")
        self.best_model_path = os.path.join("./checkpoints/", "best_model.pt")

        # Set tokenizer cleanup behavior explicitly
        self.tokenizer.clean_up_tokenization_spaces = True
        # Add padding token to the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dataset preparation
        wiki_data_train = load_dataset("wikitext", "wikitext-2-v1", split="train")
        wiki_data_val = load_dataset("wikitext", "wikitext-2-v1", split="validation")
        train_texts = [text for text in wiki_data_train["text"] if text.strip()]
        val_texts = [text for text in wiki_data_val["text"] if text.strip()]
        self.train_dataloader = self.prepare_dataloader(train_texts)
        self.val_dataloader = self.prepare_dataloader(val_texts, shuffle=False)

        # wandb.init(project="pythia-finetune", entity="wandb")

    def prepare_dataloader(self, texts, batch_size=None, shuffle=True):
        """
        Creates an efficient dataloader that handles tokenization in batches.

        Args:
            texts (List[str]): List of text samples
            batch_size (int, optional): Batch size, defaults to self.batch_size
            shuffle (bool): Whether to shuffle the dataset

        Returns:
            DataLoader: PyTorch DataLoader with dynamic batching
        """
        if batch_size is None:
            batch_size = self.batch_size

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer):
                self.texts = texts
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx]

        def collate_fn(batch):
            # Tokenize the batch of texts together for optimal padding
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }

        dataset = TextDataset(texts, self.tokenizer)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=self.device == "cuda",
            drop_last=True,
        )

    def train(self, cluster_dict, num_epochs=2, lr=5e-5):
        path = "./checkpoints/"
        os.makedirs(path, exist_ok=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        cluster_metrics = {
            "train_losses": [],
            "cluster_losses": [],
            "val_losses": [],
        }

        for epoch in range(num_epochs):
            train_metrics = self._train_epoch(
                cluster_dict, optimizer, self.num_clusters
            )
            val_metrics = self._validate_epoch(cluster_dict, self.num_clusters)
            # Store metrics for this cluster
            cluster_metrics["train_losses"].append(train_metrics["loss"])
            cluster_metrics["cluster_losses"].append(train_metrics["cluster_loss"])
            cluster_metrics["val_losses"].append(val_metrics["loss"])
            print(
                f"Epoch: {epoch}, Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}"
            )
            self._save_checkpoint(
                epoch,
                optimizer,
                train_metrics,
                val_metrics,
                num_clusters=self.num_clusters,
            )
        return cluster_metrics

    def _train_epoch(self, cluster_dict, optimizer, num_clusters):
        lmd = -40.0
        self.model.train()
        total_loss = 0
        total_cluster_loss = 0

        progress_bar = tqdm(self.train_dataloader)

        for batch_idx, batch in enumerate(progress_bar):
            input_ids, attention_mask = (
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            train_loss = outputs.loss
            cluster_loss = self._compute_cluster_loss(cluster_dict, num_clusters)
            loss = train_loss + lmd * cluster_loss  # Total loss (CE + enmeshment loss)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += train_loss.item()
            total_cluster_loss += cluster_loss.item()

            if batch_idx % self.val_freq == 0:
                val_metrics = self._validate_epoch(cluster_dict, num_clusters)

                # Check if this is the best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    # Save best model
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_metrics["loss"],
                        },
                        self.best_model_path,
                    )

                self.model.train()  # Switch back to training mode
                progress_bar.set_postfix(
                    {
                        "train_loss": f"{train_loss.item():.4f}",
                        "clusterability": f"{cluster_loss.item():.4f}",
                        "val_loss": f"{val_metrics['loss']:.4f}",
                        "best_val_loss": f"{self.best_val_loss:.4f}",
                    }
                )
            else:
                progress_bar.set_postfix(
                    {
                        "train_loss": f"{train_loss.item():.4f}",
                        "clusterability": f"{cluster_loss.item():.4f}",
                    }
                )

        return {
            "loss": total_loss / len(self.train_dataloader),
            "cluster_loss": total_cluster_loss / len(self.train_dataloader),
        }

    @torch.no_grad()
    def _validate_epoch(self, cluster_dict, num_clusters):
        self.model.eval()
        total_loss = 0
        total_cluster_loss = 0

        for batch in self.val_dataloader:
            input_ids, attention_mask = (
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            val_loss = outputs.loss
            cluster_loss = self._compute_cluster_loss(cluster_dict, num_clusters)

            total_loss += val_loss.item()
            total_cluster_loss += cluster_loss.item()

        return {
            "loss": total_loss / len(self.val_dataloader),
            "cluster_loss": total_cluster_loss / len(self.val_dataloader),
        }

    def _compute_cluster_loss(self, cluster_dict, num_clusters):
        blocks_to_cluster = [
            self.model.gpt_neox.layers[
                layer_idx
            ].mlp.dense_h_to_4h.weight  # This is pythia-specific, ideally should be abstracted
            for layer_idx in range(self.num_layers)
        ]

        UVs = [cluster_dict[i] for i in range(self.num_layers)]
        return sum(
            clusterability(block, X[0], X[1], num_clusters)
            for block, X in zip(blocks_to_cluster, UVs)
        ) / len(blocks_to_cluster)

    def _save_checkpoint(
        self, epoch, optimizer, train_metrics, val_metrics, num_clusters
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "num_clusters": num_clusters,
        }
        torch.save(
            checkpoint, f"./checkpoints/model_cluster_{num_clusters}_epoch_{epoch}.pt"
        )
