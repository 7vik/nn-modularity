import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm


###############################################################################
# Fast Flattening Utility
###############################################################################
def flatten_data(tensor: torch.Tensor) -> torch.Tensor:
    """
    Efficiently flatten a [B, S, D] activation tensor into [B*S, D].
    Ensures the result is contiguous in memory, which can speed up downstream ops.
    """
    # .flatten(0, 1) merges the first two dimensions into one
    return tensor.flatten(0, 1).contiguous()


###############################################################################
# Mutual Information (MI)
# Simple 1D → 1D histogram-based computation
###############################################################################
def mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 30) -> float:
    """
    Estimate pairwise Mutual Information between x and y, each shape [N].
    Returns MI in bits (log base 2).
    """
    x_np = x if isinstance(x, np.ndarray) else x.cpu().numpy()
    y_np = y if isinstance(y, np.ndarray) else y.cpu().numpy()

    # 2D histogram for joint
    joint_hist2d, xedges, yedges = np.histogram2d(x_np, y_np, bins=bins, density=False)
    joint_prob = joint_hist2d / joint_hist2d.sum()
    # Marginal distributions
    x_prob = np.sum(joint_prob, axis=1)
    y_prob = np.sum(joint_prob, axis=0)

    mi_val = 0.0
    for i in range(len(x_prob)):
        for j in range(len(y_prob)):
            if joint_prob[i, j] > 0.0:
                mi_val += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (x_prob[i] * y_prob[j])
                )
    return float(mi_val)


###############################################################################
# HSIC (Hilbert-Schmidt Independence Criterion) using an RBF kernel
###############################################################################
def rbf_kernel(array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute an RBF kernel matrix for array [N, d].
    Returns [N, N].
    """
    if array.ndim == 1:
        array = array[:, None]  # make it 2D
    XX = np.sum(array**2, axis=1, keepdims=True)  # [N,1]
    D = XX - 2 * array.dot(array.T) + XX.T
    K = np.exp(-D / (2 * sigma**2))
    return K


def hsic(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    """
    Estimate HSIC for two arrays x and y, each shape [N] or [N, d].
    """
    # Convert to numpy
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

    Kx = rbf_kernel(x_np, sigma)
    Ky = rbf_kernel(y_np, sigma)

    n = Kx.shape[0]
    H = np.eye(n) - (1.0 / n) * np.ones((n, n))
    KxH = Kx.dot(H)
    KyH = Ky.dot(H)
    val = np.trace(KxH.dot(KyH)) / ((n - 1) ** 2)
    return float(val)


###############################################################################
# Covariance, cluster means, and subsequent plotting
###############################################################################
def compute_covariances(batch_2d: torch.Tensor):
    """
    batch_2d: [N, 3072] or similar
    Returns:
      cov         : [D, D] covariance
      mean_cov    : [4, 4] covariance among 4 cluster means
      cluster_means : [N, 4] each row is the mean of that example's chunk
    """
    # Center
    batch_2d = batch_2d.to("cuda", non_blocking=True)
    mean = batch_2d.mean(dim=0, keepdim=True)
    centered = batch_2d - mean

    N = centered.size(0)
    # Big matmul on GPU
    cov = (centered.T @ centered) / (N - 1)  # shape [3072, 3072]

    # Split into 4 clusters for cluster_means, etc.
    d = batch_2d.size(1)
    step = d // 4
    c_means = []
    for i in range(4):
        start = i * step
        end = (i + 1) * step if i < 3 else d
        c_means.append(centered[:, start:end].mean(dim=1, keepdim=True))
    cluster_means = torch.cat(c_means, dim=1)  # [N,4]
    mean_cov = (cluster_means.T @ cluster_means) / (N - 1)

    # Move results back to CPU for plotting etc.
    cov_cpu = cov.cpu()
    mean_cov_cpu = mean_cov.cpu()
    cluster_means_cpu = cluster_means.cpu()
    return cov_cpu, mean_cov_cpu, cluster_means_cpu


def plot_cov(cov: torch.Tensor, mean_cov: torch.Tensor, layer_idx: int, out_dir: str):
    # Full covariance can be huge, but we'll still do a heatmap.
    # For 3072 x 3072, that might be large to display.
    # If needed for debugging or a smaller model, it is here:
    plt.figure(figsize=(10, 10))
    sns.heatmap(cov.cpu().numpy(), cmap="hot", square=True, cbar=True, annot=False)
    plt.title(f"Layer {layer_idx} - Full Cov Matrix")
    plt.savefig(os.path.join(out_dir, f"covariance_matrix_{layer_idx}.png"))
    plt.close()

    # 4x4 cluster covariance
    plt.figure(figsize=(4, 4))
    sns.heatmap(mean_cov.cpu().numpy(), cmap="hot", annot=True, fmt=".2f", square=True)
    plt.title(f"Layer {layer_idx} - 4-Cluster Cov")
    plt.savefig(os.path.join(out_dir, f"mean_covariance_matrix_{layer_idx}.png"))
    plt.close()


###############################################################################
# Activation Pattern Analysis
###############################################################################
def plot_activation_patterns(cluster_means: torch.Tensor, layer_idx: int, out_dir: str):
    """
    cluster_means: [N, 4]
    We'll do a boxplot of each cluster's normalized usage across examples.
    """
    with torch.no_grad():
        # shift to ensure positivity if negative
        min_val = cluster_means.min().item()
        offset = 0.0
        if min_val < 0:
            offset = -min_val + 1e-5
        shifted = cluster_means + offset  # shape [N,4]
        row_sums = shifted.sum(dim=1, keepdim=True) + 1e-9
        pattern = shifted / row_sums  # shape [N,4]

    # Subsample to 1k
    n = pattern.size(0)
    sample_size = min(n, 1000)
    idxs = torch.randperm(n)[:sample_size]
    pattern_sample = pattern[idxs].cpu().numpy()

    import pandas as pd

    df = pd.DataFrame(pattern_sample, columns=[f"C{i}" for i in range(4)])
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df)
    plt.title(f"Layer {layer_idx} - Normalized Activation Patterns")
    plt.savefig(os.path.join(out_dir, f"activation_patterns_{layer_idx}.png"))
    plt.close()


###############################################################################
# The Main Analysis Loop: Minimizing Overhead
###############################################################################
def activation_analysis():
    """
    Loads chunk0_*.pt files, flatten them, and performs:
      1) Covariance across entire [N, D]
      2) Cluster 4-split, cluster covariance
      3) Pairwise MI & HSIC among the 4 cluster means
      4) Activation pattern analysis
    Does so layer by layer to avoid huge memory usage.
    """
    data_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/interp-gains-gpt2small/data_refactored"
    out_dir = (
        "/data/joan_velja/nn-modularity/interp-gains-gpt2small/plots_v2_correlations"
    )
    os.makedirs(out_dir, exist_ok=True)

    # We'll only process files starting with "chunk0"
    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith("chunk0")])
    num_layers = 12

    for layer_idx in range(num_layers):
        print(f"\nProcessing layer {layer_idx}...")

        # Keep flattened chunks in a list, then cat once
        layer_data_list = []

        for fname in tqdm(file_list, desc=f"Layer {layer_idx}"):
            path = os.path.join(data_dir, fname)
            data_dict = torch.load(path)  # -> {layer_idx: [B, S, D]}
            if layer_idx in data_dict:
                data_cpu = data_dict[layer_idx]  # shape [B,S, D]
                # Flatten => [B*S, D]
                data_flat = flatten_data(data_cpu)
                layer_data_list.append(data_flat)
            del data_dict
            torch.cuda.empty_cache()

        if not layer_data_list:
            print(f"No data found for layer {layer_idx}. Skipping.")
            continue

        # Single concat instead of repeated cat => big speed gain
        full_data = torch.cat(layer_data_list, dim=0)  # shape [N, D]
        del layer_data_list

        # (A) Covariance + 4 cluster means
        cov, mean_cov, cluster_means = compute_covariances(full_data)
        plot_cov(cov, mean_cov, layer_idx, out_dir)

        # # (B) Pairwise MI among the 4 cluster vectors
        # #     We'll convert cluster_means to numpy once to avoid repeated CPU calls
        # cluster_np = cluster_means.cpu().numpy()  # shape [N,4]
        # mi_mat = np.zeros((4,4), dtype=np.float32)
        # for i in range(4):
        #     for j in range(i+1, 4):
        #         x = cluster_np[:, i]
        #         y = cluster_np[:, j]
        #         mi_val = mutual_information(x, y, bins=30)
        #         mi_mat[i,j] = mi_val
        #         mi_mat[j,i] = mi_val

        # plt.figure(figsize=(4,4))
        # sns.heatmap(mi_mat, annot=True, cmap="Blues", fmt=".2f", square=True)
        # plt.title(f"Layer {layer_idx} - MI among 4 clusters")
        # plt.savefig(os.path.join(out_dir, f"mi_matrix_{layer_idx}.png"))
        # plt.close()

        # # (C) Pairwise HSIC among the 4 cluster vectors
        # hsic_mat = np.zeros((4,4), dtype=np.float32)
        # for i in range(4):
        #     for j in range(i+1, 4):
        #         x = cluster_np[:, i]
        #         y = cluster_np[:, j]
        #         h_val = hsic(x, y, sigma=1.0)
        #         hsic_mat[i,j] = h_val
        #         hsic_mat[j,i] = h_val

        # plt.figure(figsize=(4,4))
        # sns.heatmap(hsic_mat, annot=True, cmap="Greens", fmt=".3f", square=True)
        # plt.title(f"Layer {layer_idx} - HSIC among 4 clusters")
        # plt.savefig(os.path.join(out_dir, f"hsic_matrix_{layer_idx}.png"))
        # plt.close()

        # # (D) Activation pattern analysis
        # plot_activation_patterns(cluster_means, layer_idx, out_dir)

        # Free memory
        del full_data
        del cluster_means
        torch.cuda.empty_cache()

    print("\nAll layers processed successfully.")


def main():
    activation_analysis()


if __name__ == "__main__":
    main()
