import json
import os

import torch
import wandb
from cluster import Clusters
from datasets import load_dataset
from tqdm import tqdm
from utils import clusterability, get_device, get_mlp_parameters


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        num_clusters,
        model_name,
        ckpt_name,
        steps_to_cluster=0,
        enable_BSGC=False,
        do_modularity=False,
        mix_ravel=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = self.model.config.num_hidden_layers
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.device = get_device()
        self.steps = 0
        self.val_freq = 100
        self.steps_to_cluster = steps_to_cluster
        self.best_val_loss = float("inf")
        self.model_name = model_name
        self.ckpt_name = ckpt_name
        self.best_model_path = os.path.join(
            f"./checkpoints_{ckpt_name}/", "best_model.pt"
        )
        path = f"./checkpoints_{ckpt_name}/"
        os.makedirs(path, exist_ok=True)

        self.cluster_dict = None
        self.enable_BSGC = enable_BSGC
        self.do_modularity = do_modularity
        self.mix_ravel = mix_ravel

        print(f"Model name: {self.model_name}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Steps to cluster: {self.steps_to_cluster}")
        print(f"Enable BSGC: {self.enable_BSGC}")
        print(f"Do modularity: {self.do_modularity}")


        # Set tokenizer cleanup behavior explicitly
        self.tokenizer.clean_up_tokenization_spaces = True
        # Add padding token to the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dataset preparation
        wiki_data_train = load_dataset("wikitext", "wikitext-2-v1", split="train")
        wiki_data_val = load_dataset("wikitext", "wikitext-2-v1", split="validation")

        if self.mix_ravel:
            print("Mixing RAVEL data with the training data")
            # Add RAVEL to the training data
            with open("/data/joan_velja/nn-modularity/language-models/pythia-finetune/gpt2_prompt_data.json", "r") as f:
                ravel_data = json.load(f)
                # ravel_data is a dict of (entity-target) key and list((prompt, response)) value
                # Extract the prompts and responses
                ravel_texts = [
                    item[0] + ' ' + item[1] for sublist in ravel_data.values() for item in sublist
                ]
            

        train_texts = [text for text in wiki_data_train["text"] if text.strip()]
        val_texts = [text for text in wiki_data_val["text"] if text.strip()]

        if self.mix_ravel: # Mix RAVEL data with the training data
            train_texts += ravel_texts
    
        self.train_dataloader = self.prepare_dataloader(train_texts)
        self.val_dataloader = self.prepare_dataloader(val_texts, shuffle=False)

        # wandb.init(project="pythia-finetune", entity="wandb")
        wandb.init(project="pythia-finetune")
        wandb.config.update(
            {
                "model_name": self.model.config.name_or_path,
                "batch_size": self.batch_size,
                "num_clusters": self.num_clusters,
                "steps_to_cluster": self.steps_to_cluster,
                "enable_BSGC": self.enable_BSGC,
                "do_modularity": self.do_modularity,
            }
        )

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

    def train(self, cluster_dict, num_epochs=2, lr=5e-7):
        self.model.train()
        self.steps = 0
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
            cluster_loss = torch.tensor(0.0)

            if self.steps == self.steps_to_cluster and self.do_modularity:
                # Recluster the model
                self.cluster_dict = Clusters(
                    self.model,
                    self.tokenizer,
                    num_clusters,
                    enable_BSGC=self.enable_BSGC,
                ).forward()
                print("Model has been reclustered.")

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
            if self.do_modularity and self.steps > self.steps_to_cluster:
                cluster_loss = self._compute_cluster_loss(self.cluster_dict, num_clusters)
            loss = train_loss + lmd * cluster_loss

            loss.backward()
            grads = [
                param.grad.view(-1)
                for param in self.model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.norm(torch.cat(grads), 2).item() if grads else 0.0

            optimizer.step()
            optimizer.zero_grad()

            total_loss += train_loss.item()
            total_cluster_loss += (
                cluster_loss.item() if cluster_loss else torch.tensor(0.0)
            )

            # Log to wandb
            wandb.log(
                {
                    "train_loss": train_loss.item(),
                    "cluster_loss": cluster_loss.item(),
                    "grad_norm": grad_norm,
                },
                step=self.steps,
            )

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
                        "grad_norm": f"{grad_norm:.4f}",
                    }
                )
            else:
                progress_bar.set_postfix(
                    {
                        "train_loss": f"{train_loss.item():.4f}",
                        "clusterability": f"{cluster_loss.item():.4f}",
                        "grad_norm": f"{grad_norm:.4f}",
                    }
                )
            self.steps += 1

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
            cluster_loss = torch.tensor(0.0)
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
            if self.do_modularity and self.steps > self.steps_to_cluster:
                cluster_loss = self._compute_cluster_loss(cluster_dict, num_clusters)

            total_loss += val_loss.item()
            total_cluster_loss += cluster_loss.item()

        return {
            "loss": total_loss / len(self.val_dataloader),
            "cluster_loss": total_cluster_loss / len(self.val_dataloader),
        }

    def _compute_cluster_loss(self, cluster_dict, num_clusters):
        # blocks_to_cluster = [
        #     self.model.gpt_neox.layers[
        #         layer_idx
        #     ].mlp.dense_h_to_4h.weight  # This is pythia-specific, ideally should be abstracted
        #     for layer_idx in range(self.num_layers)
        # ]
        blocks_to_cluster = get_mlp_parameters(self.model, "in")

        UVs = [cluster_dict[i] for i in range(self.num_layers)]
        return sum(
            clusterability(block, X[0], X[1], num_clusters, is_gpt="gpt" in self.model_name)
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
            checkpoint,
            f"./checkpoints_{self.ckpt_name}/model_cluster_{num_clusters}_epoch_{epoch}.pt",
        )
