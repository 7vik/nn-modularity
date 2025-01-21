from __init__ import *

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def activation_analysis():
    # Input directory: Where .pt files reside (e.g. chunk0_batchX.pt)
    data_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/interp-gains-gpt2small/data_refactored"
    # Output directory: Where we save plots
    out_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/plots"
    os.makedirs(out_dir, exist_ok=True)

    # We only process files whose names start with "chunk0" (per your original code).
    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith("chunk0")])

    # --------------------------------------------------------------------------
    # Helper function: compute covariance matrices of the raw neuronal norms
    # and the 4-cluster aggregated mean activations (as in original code).
    # --------------------------------------------------------------------------
    def covariance_matrix(batch_):
        """
        batch_: shape [N, 1024, seq_len], the concatenated activations for one layer
        Returns:
          cov       : [1024 x 1024] covariance
          mean_cov  : [4 x 4] covariance of the 4 cluster means
        """
        # 1) L2 norm across the last dimension => shape [bs, 1024 = seq_len, 3072 = d_model] -> [bs, 1024]
        print("batch_ shape:", batch_.shape)
        values = torch.norm(batch_, dim=-1)  # shape: [N] if batch_ is [N, 1024]
        # 2) Center the data (1D)
        mean = values.mean(dim=0)
        centered = values - mean  # shape: [N]

        # Alternative: (bs, seq_len, d_model) tensor
        # 1) flatten over 0,1 --> N = bs x seq_len --> 2D tensor (N, d_model) --> cov outer product 
        # N likely very big (128 * 1024)
        # matmul implies (centered.T, centered) --> N x N x 3072

        # Nandi's suggestion: 
        # batch_ = batch_[:, -1, :] shape : (bs = 128, 1, 3072)

        # cov = (centered.T, centered) = ((3072,128) x (128, 3072)) = (3072,3072)

        # Sampling from bs=128
        
        # 3) Split into clusters
        cluster1 = torch.mean(centered[:, :256], dim=1).reshape(-1, 1)
        cluster2 = torch.mean(centered[:, 256:512], dim=1).reshape(-1, 1)
        cluster3 = torch.mean(centered[:, 512:768], dim=1).reshape(-1, 1)
        cluster4 = torch.mean(centered[:, 768:], dim=1).reshape(-1, 1)
        all_cluster = torch.cat([cluster1, cluster2, cluster3, cluster4], dim=1)
        # 4) Full covariance
        cov = torch.matmul(centered.T, centered) / (centered.size(0) - 1)  # shape: [1024, 1024] --> ideal: [3072, 3072]
        # 5) Mean covariance (between the 4 clusters)
        mean_cov = torch.matmul(all_cluster.T, all_cluster) / (centered.size(0) - 1)
        return cov, mean_cov

    # --------------------------------------------------------------------------
    # For plotting both the full covariance and the cluster covariance
    # --------------------------------------------------------------------------
    def visualize_and_save(batch_, layer_idx):
        cov, mean_cov = covariance_matrix(batch_)
        plt.figure(figsize=(10,10))
        sns.heatmap(cov.cpu().numpy(), cmap="hot", annot=False)
        plt.title(f"Layer {layer_idx} - Full Covariance")
        plt.savefig(os.path.join(out_dir, f"covariance_matrix_{layer_idx}.png"))
        plt.close()

        plt.figure(figsize=(6,6))
        sns.heatmap(mean_cov.cpu().numpy(), cmap='hot', annot=True, fmt='g')
        plt.title(f"Layer {layer_idx} - 4-Cluster Covariance")
        plt.savefig(os.path.join(out_dir, f"mean_covariance_matrix_{layer_idx}.png"))
        plt.close()
    
    # --------------------------------------------------------------------------
    # Process each layer separately to avoid storing everything in memory at once
    # --------------------------------------------------------------------------
    def process_single_layer(layer_idx):
        # Collect all batch data for this layer
        layer_data_list = []
        for fname in file_list:
            path = os.path.join(data_dir, fname)
            loaded_dict = torch.load(path)  # shape: {layer_idx -> [num_examples, 1024]}
            if layer_idx in loaded_dict:
                # Move to CPU to avoid GPU memory blowups
                data_cpu = loaded_dict[layer_idx].cpu()
                print(f'data_cpu shape: {data_cpu.shape}')
                layer_data_list.append(data_cpu)
            del loaded_dict
            torch.cuda.empty_cache()
        
        # Concatenate all for this layer
        if len(layer_data_list) == 0:
            print(f"No data found for layer {layer_idx}, skipping.")
            return
        
        full_data = torch.cat(layer_data_list, dim=0)  # shape e.g. [N_total, 1024]
        del layer_data_list
        
        # Now run the same analysis you had in "visualize(...)"
        visualize_and_save(full_data, layer_idx)
        
        # Free memory
        del full_data
        torch.cuda.empty_cache()

    # Actual loop: For each layer, gather data from all chunk0 files, do the analysis, discard
    for layer_idx in range(12):
        print(f"Processing layer {layer_idx} ...")
        process_single_layer(layer_idx)
        print(f"Done layer {layer_idx}")

def main():
    activation_analysis()

if __name__ == "__main__":
    main()