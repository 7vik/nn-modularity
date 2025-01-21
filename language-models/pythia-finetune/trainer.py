import os

import torch
import transformer_lens.utils as tutils
from datasets import Dataset, load_dataset
from torch.amp import GradScaler, autocast
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
        self.scaler = GradScaler("cuda")

        # Set tokenizer cleanup behavior explicitly
        self.tokenizer.clean_up_tokenization_spaces = True

        # Dataset preparation
        train_ds, val_ds = self.prep_dataset()
        self.train_dataloader = self.get_dataloader(train_ds, shuffle=True)
        self.val_dataloader = self.get_dataloader(val_ds, shuffle=False)
        # wandb.init(project="pythia-finetune", entity="wandb")

    def get_dataloader(self, dataset, shuffle):
        """
        Create a PyTorch DataLoader object from the dataset

        Args:
            dataset (dict): Dictionary containing the input_ids and attention_mask
            shuffle (bool): Whether to shuffle the dataset

        Returns:
            DataLoader: PyTorch DataLoader object

        """
        return torch.utils.data.DataLoader(
            Dataset.from_dict(dataset),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
            if self.device == "cuda"
            else False,  # Not supported on Apple Silicon
        )

    def prep_dataset(self):
        """
        Load and tokenize the datasets for training and validation. Currently using wikitext-2 dataset (ideally should be a parameter)

        Returns:
            dict: Dictionary containing the input_ids and attention_mask for training and validation datasets

        """
        # Load dataset
        wiki_data_train = load_dataset("wikitext", "wikitext-2-v1", split="train")
        wiki_data_val = load_dataset("wikitext", "wikitext-2-v1", split="validation")

        return self._process_split(wiki_data_train), self._process_split(wiki_data_val)

    def _process_split(self, dataset):
        """
        Tokenize and concatenate the text data in the dataset

        Args:
            dataset (Dataset): Huggingface dataset object

        Returns:
            dict: Dictionary containing the input_ids and attention_mask

        """
        texts = [text for text in dataset["text"] if text.strip()]
        tokenized_dataset = tutils.tokenize_and_concatenate(
            Dataset.from_dict({"text": texts}),
            self.tokenizer,
            streaming=False,
            max_length=512,
            column_name="text",
            add_bos_token=True,
            num_proc=10,
        )

        return {
            "input_ids": tokenized_dataset["tokens"],
            "attention_mask": torch.ones_like(tokenized_dataset["tokens"]),
        }

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

    @torch.amp.autocast(device_type="cuda")
    def _train_epoch(self, cluster_dict, optimizer, num_clusters):
        lmd = -40.0
        self.model.train()
        total_loss = 0
        total_cluster_loss = 0

        progress_bar = tqdm(self.train_dataloader)

        for batch in progress_bar:
            with autocast(device_type="cuda"):
                input_ids, attention_mask = [t.to(self.device) for t in batch]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                train_loss = outputs.loss
                cluster_loss = self._compute_cluster_loss(cluster_dict, num_clusters)
                loss = (
                    train_loss + lmd * cluster_loss
                )  # Total loss (CE + enmeshment loss)

            # Gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)

            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()

            total_loss += train_loss.item()
            total_cluster_loss += cluster_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_loss.item():.4f}",
                    "cluster_loss": f"{cluster_loss.item():.4f}",
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
            input_ids, attention_mask = [t.to(self.device) for t in batch]

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
