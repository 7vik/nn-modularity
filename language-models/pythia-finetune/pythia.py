import argparse
import pickle as pkl

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import clusterability, get_device, set_all_seeds, spectral_clustering
from transformer_lens.evals import make_wiki_data_loader


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

    def train(self, cluster_dict):
        cluster_losses = []
        train_losses = []
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

        # cluster_dict[0]

        for epoch in range(num_epochs):
            wiki = make_wiki_data_loader(self.tokenizer, batch_size=self.batch_size)
            for idx, batch in enumerate(wiki.dataset['tokens']):
                tokens = batch["tokens"].to(self.device)

                # CLUSTERABILITY LOSS
                UVs = [
                    cluster_dict[i] for i in keys
                ]  # list of tuples of U, V for each layer (len = num_layers)
                cluster_loss = sum(
                    [
                        clusterability(block, U, V)
                        for (block, U, V) in zip(blocks_to_cluster, UVs)
                    ]
                ) / len(blocks_to_cluster)

                train_loss = self.model(tokens, return_type="loss")
                cluster_losses.append(cluster_loss.item())
                train_losses.append(train_loss.item())
                loss = train_loss - lomda * cluster_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch {idx}, Train Loss: {round(train_loss.item(), 4)}, Clusterability: {round(cluster_loss, 4)}"
                    )
            torch.save(
                self.model.state_dict(),
                path + f"wiki_non_modular_mlp_in_model_epoch_{epoch + 1}.pt",
            )

        # store the cluster losses and train losses
        with open(path + "wiki_non_modular_mlp_in_cluster_losses.pkl", "wb") as f:
            pkl.dump(cluster_losses, f)
        with open(path + "wiki_non_modular_mlp_in_train_losses.pkl", "wb") as f:
            pkl.dump(train_losses, f)


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
