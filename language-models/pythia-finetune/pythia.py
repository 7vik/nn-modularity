import argparse
import os
import pickle as pkl

import torch
import transformer_lens.utils as utils

# Dataset is from huggingface/datasets
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import clusterability, get_device, set_all_seeds, spectral_clustering


class Config:
    def __init__(self):
        self.model_name = "EleutherAI/pythia-70m"

    def config(self) -> tuple[AutoModelForCausalLM, AutoTokenizer, list[int]]:
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        cluster_set = list(range(2, 21))
        return model, tokenizer, cluster_set

    def forward(self):
        set_all_seeds(42, warn_only=True)
        model, tokenizer, num_clusters = self.config()
        return model, tokenizer, num_clusters


class Clusters:
    def __init__(self, model, tokenizer, num_clusters):
        self.model = model
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters
        self.num_layers = self.model.config.num_hidden_layers

    def forward(self):
        svd_cluster_dict = {cluster_idx: {} for cluster_idx in self.num_clusters}
        for cluster in tqdm(self.num_clusters):
            for layer_idx in range(self.num_layers):
                U, V = spectral_clustering(
                    self.model.gpt_neox.layers[
                        layer_idx
                    ].mlp.dense_h_to_4h.weight,  # This is pythia-specific, ideally should be abstracted
                    cluster,
                )
                svd_cluster_dict[cluster][layer_idx] = (U, V)

        with open("svd_dict.pkl", "wb") as f:
            pkl.dump(svd_cluster_dict, f)

        print("SVD clustering done!")
        print("Saved the SVD dictionary to svd_dict.pkl")
        return svd_cluster_dict


class Trainer:
    def __init__(self, model, tokenizer, num_clusters, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters
        self.num_layers = self.model.config.num_hidden_layers
        self.batch_size = batch_size
        self.device = get_device()
        # CE loss
        self.loss = torch.nn.CrossEntropyLoss()
        # Dataset for training
        self.prep_dataset()

    def prep_dataset(self):
        # Load dataset
        wiki_data = load_dataset("wikitext", "wikitext-2-v1", split="train")
        texts = wiki_data["text"]

        # Filter empty strings and get non-empty texts
        texts = [text for text in texts if text.strip()]

        # Use tokenize_and_concatenate for efficient tokenization and padding
        tokenized_dataset = utils.tokenize_and_concatenate(
            Dataset.from_dict({"text": texts}),
            self.tokenizer,
            streaming=False,
            max_length=512,  # Standard context length
            column_name="text",
            add_bos_token=True,  # Add beginning of sequence token
            num_proc=10,  # Parallel processing for speed
        )

        # Convert to tensor format
        self.wiki_dataset = {
            "input_ids": tokenized_dataset["tokens"],
            "attention_mask": torch.ones_like(tokenized_dataset["tokens"]),
        }

    def train(self, cluster_dict):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        train_losses, cluster_losses = [], []
        lomda = 40.0  # Feels iffy
        blocks_to_cluster = [
            self.model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.weight
            for layer_idx in range(self.num_layers)
        ]  # This is pythia-specific, ideally should be abstracted
        path = "./checkpoints/"
        num_epochs = 2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        keys = list(range(self.num_layers))  # keys of cluster_dict

        # Create proper DataLoader
        train_dataloader = DataLoader(
            TensorDataset(
                self.wiki_dataset["input_ids"], self.wiki_dataset["attention_mask"]
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Added the loop for each cluster.
        for cluster in self.num_clusters:
            for epoch in range(num_epochs):
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
                epoch_losses = []
                # for idx, batch in enumerate(self.wiki_dataset["tokens"]):
                for batch in progress_bar:
                    # TODO: the loss cannot be computed as written in 101, as it is not hookedtransformer

                    input_ids, attention_mask = [t.to(self.device) for t in batch]

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,  # For causal LM loss
                    )
                    train_loss = outputs.loss

                    # CLUSTERABILITY LOSS
                    UVs = [
                        cluster_dict[i] for i in keys
                    ]  # list of tuples of U, V for each layer (len = num_layers)
                    cluster_loss = sum(
                        [
                            clusterability(block, X[0], X[1], cluster)
                            for (block, X) in zip(blocks_to_cluster, UVs)
                        ]
                    ) / len(blocks_to_cluster)

                    # Combined loss
                    loss = train_loss - lomda * cluster_loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track losses
                    train_losses.append(train_loss.item())
                    cluster_losses.append(cluster_loss.item())
                    epoch_losses.append(loss.item())

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "train_loss": f"{train_loss.item():.4f}",
                            "clusterability": f"{cluster_loss.item():.4f}",
                        }
                    )

                # # store the cluster losses and train losses
                # with open(
                #     path + f"wiki_non_modular_mlp_in_cluster{cluster}_losses.pkl", "wb"
                # ) as f:
                #     pkl.dump(cluster_losses, f)
                # with open(
                #     path + f"wiki_non_modular_mlp_in_cluster{cluster}_train_losses.pkl",
                #     "wb",
                # ) as f:
                #     pkl.dump(train_losses, f)
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        path, f"model_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pt"
                    ),
                )
                print(f"Model saved at epoch {epoch + 1} with loss {avg_loss:.4f}")
                print("===" * 10)

        return train_losses, cluster_losses


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    c = Config()
    model, tokenizer, num_clusters = c.forward()
    # cl = Clusters(model, tokenizer, num_clusters)
    # svd = cl.forward()

    trainer = Trainer(model, tokenizer, num_clusters, batch_size=args.batch_size)

    with open("svd_dict.pkl", "rb") as f:
        svd_dict = pkl.load(f)

    for (
        cluster_value,
        cluster_dict,
    ) in svd_dict.items():  # Loop through num_clusters (keys of svd_dict) noqa
        # for key, value in svd_dict[cluster_value].items():
        #     # pprint(value)
        #     U, V = value
        # break
        print(f"C: {cluster_value}")
        print(
            f"CD: {cluster_dict.keys()}"
        )  # cluster_dict.keys() is a list of layer indices, each containing U, V indices pair for that layer
        print("---" * 10)
        # cluster_dict[0] == tuple(U, V)

        trainer.train(cluster_dict)


if __name__ == "__main__":
    main()
