import numpy as np
from scipy.sparse.linalg import svds
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
import argparse
import pickle as pkl
from tqdm import tqdm
from pprint import pprint


class config:

    def __init__ (self):
        self.model_name = "EleutherAI/pythia-70m"

    def set_all_seeds(self, seed, deterministic=True):
        """
        Set all seeds and deterministic flags for reproducibility.
        
        Args:
            seed (int): The seed value to use for all random number generators
            deterministic (bool): Whether to enforce deterministic behavior
        """
        # Python RNG
        random.seed(seed)
        
        # NumPy RNG
        np.random.seed(seed)
        
        # PyTorch RNGs
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Environment variables
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if deterministic:
            # CUDA deterministic behavior
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        
    def config(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        print(model)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        cluster_set = list(range(2, 21))
        return model, tokenizer, cluster_set
    
    def forward(self):
        self.set_all_seeds(42)
        model, tokenizer, num_clusters = self.config()
        return model, tokenizer, num_clusters
    


class clusters:

    def __init__(self, model, tokenizer, num_clusters):
        self.model = model
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters        

    def forward(self):
        svd_cluster_dict = {cluster_idx:{} for cluster_idx in self.num_clusters}
        for cluster in tqdm(self.num_clusters):
            for layer_idx in range(6):
                U, V = spectral_clustering(
                        self.model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.weight, 
                        cluster
                        )
                svd_cluster_dict[cluster][layer_idx] = (U, V)

        with open("svd_dict.pkl", "wb") as f:
            pkl.dump(svd_cluster_dict, f)
            


def spectral_clustering(layer, k):
    
    A = layer.detach().cpu().numpy()
    A = np.abs(A)
    
    D_U = np.diag(np.sum(A, axis=1))
    D_V = np.diag(np.sum(A, axis=0))

    D_U_inv_sqrt = np.linalg.inv(np.sqrt(D_U))
    D_V_inv_sqrt = np.linalg.inv(np.sqrt(D_V))

    A_tilde = D_U_inv_sqrt @ A @ D_V_inv_sqrt

    U, Sigma, Vt = svds(A_tilde, k=k)

    kmeans_U = KMeans(n_clusters=k, random_state=42).fit(U)
    kmeans_V = KMeans(n_clusters=k, random_state=42).fit(Vt.T)

    labels_U = kmeans_U.labels_
    labels_V = kmeans_V.labels_

    # convert labels to indices
    cluster_U_indices = defaultdict(list)
    cluster_V_indices = defaultdict(list)
    for i, label in enumerate(labels_U):
        cluster_U_indices[label].append(i)
    for i, label in enumerate(labels_V):
        cluster_V_indices[label].append(i)

    return cluster_U_indices, cluster_V_indices



def cluster_goodness_fast(model, cluster_U_indices, cluster_V_indices, num_clusters):
    A = model.fc2.weight ** 2
    mask = torch.zeros_like(A, dtype=torch.bool)
    
    for cluster_idx in range(num_clusters):
        u_indices = torch.tensor(cluster_U_indices[cluster_idx], dtype=torch.long)
        v_indices = torch.tensor(cluster_V_indices[cluster_idx], dtype=torch.long)
        mask[u_indices.unsqueeze(1), v_indices] = True
    
    intra_cluster_out_sum = torch.sum(A[mask])
    total_out_sum = torch.sum(A)
    
    return intra_cluster_out_sum / total_out_sum


def train():
    cluster_losses = []
    train_losses = []
    lomda = 40.0
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.to(device)
    blocks_to_cluster = [model.blocks[i].mlp.W_in for i in range(12)]
    path = './checkpoints/'
    num_epochs = 2
    num_clusters = 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
        
    for epoch in range(num_epochs):
        for idx, batch in enumerate(datasets['wiki']):
            tokens = batch['tokens'].to(device)
            # cluster_loss = sum([clusterability(block) for block in blocks_to_cluster]) / len(blocks_to_cluster)
            cluster_loss = 0
            train_loss = model(tokens, return_type="loss")
            # cluster_losses.append(cluster_loss.item())
            cluster_losses.append(0)
            train_losses.append(train_loss.item())
            loss = train_loss - lomda * cluster_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {idx}, Train Loss: {round(train_loss.item(), 4)}, Clusterability: {round(cluster_loss, 4)}')    
        torch.save(model.state_dict(), path + f'wiki_non_modular_mlp_in_model_epoch_{epoch+1}.pt')

    # store the cluster losses and train losses
    import pickle
    with open(path + 'wiki_non_modular_mlp_in_cluster_losses.pkl', 'wb') as f:
        pickle.dump(cluster_losses, f)
    with open(path + 'wiki_non_modular_mlp_in_train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    args = parser.parse_args()
    # c = config()
    # model, tokenizer, num_clusters = c.forward()
    # cl = clusters(model, tokenizer, num_clusters)
    # cl.forward()

    with open("svd_dict.pkl", "rb") as f:
        svd_dict = pkl.load(f)
    
    for cluster_value, cluster_dict in svd_dict.items():
        for key, value in svd_dict[cluster_value].items():
            pprint(value)
        break

if __name__ == "__main__":
    main()