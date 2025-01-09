import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1) Mutual Information (MI) function
#    - Simple histogram-based approach for two 1D vectors
# ─────────────────────────────────────────────────────────────────────────────
def mutual_information(x, y, bins=30):
    """
    Estimate pairwise Mutual Information between x and y (1D Tensors).
    Args:
        x (tensor): shape [N]
        y (tensor): shape [N]
        bins (int) : number of histogram bins
    Returns:
        mi (float): estimated mutual information in bits
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # Compute 2D joint distribution (histogram)
    joint_hist2d, xedges, yedges = np.histogram2d(x_np, y_np, bins=bins, density=False)
    joint_prob = joint_hist2d / np.sum(joint_hist2d)

    # Marginals
    x_prob = np.sum(joint_prob, axis=1)
    y_prob = np.sum(joint_prob, axis=0)

    mi_val = 0.0
    for i in range(len(x_prob)):
        for j in range(len(y_prob)):
            if joint_prob[i, j] > 0.0:
                mi_val += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
    return float(mi_val)

# ─────────────────────────────────────────────────────────────────────────────
# 2) HSIC function
#    - RBF kernel approach
# ─────────────────────────────────────────────────────────────────────────────
def rbf_kernel(X, sigma=1.0):
    """
    Compute RBF kernel matrix for X (shape [N, D]).
    Returns an (N x N) kernel matrix.
    """
    if len(X.shape) == 1:
        X = X[:, None]
    XX = np.sum(X**2, axis=1, keepdims=True)
    D = XX - 2*np.dot(X, X.T) + XX.T
    K = np.exp(-D / (2 * sigma**2))
    return K

def hsic(X, Y, sigma=1.0):
    """
    Hilbert-Schmidt Independence Criterion between two inputs X, Y (shape [N] or [N, D]).
    Returns a scalar HSIC estimate.
    """
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    Kx = rbf_kernel(X_np, sigma)
    Ky = rbf_kernel(Y_np, sigma)

    n = Kx.shape[0]
    H = np.eye(n) - (1.0 / n) * np.ones((n, n))
    # HSIC ~ trace(Kx H Ky H) / (n-1)^2 or similar constants.
    KxH = np.dot(Kx, H)
    KyH = np.dot(Ky, H)
    hsic_val = np.trace(np.dot(KxH, KyH)) / ((n - 1) ** 2)
    return float(hsic_val)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Original-style covariance & cluster means
#    Now dimension = 3072 for each token, flattened from [batch_size, seq_len].
# ─────────────────────────────────────────────────────────────────────────────
def compute_covariances(batch_2d):
    """
    batch_2d: [N, 3072]  (flattened from [batch_size*seq_len, d_model])
    Returns:
       cov: [3072 x 3072] covariance
       mean_cov: [4 x 4] covariance of 4 aggregated clusters
       cluster_means: [N, 4] the per-example cluster means
    """
    # Center
    mean = batch_2d.mean(dim=0, keepdim=True)  # shape [1, 3072]
    centered = batch_2d - mean                 # shape [N, 3072]

    # Full covariance
    N = centered.size(0)
    cov = torch.matmul(centered.T, centered) / (N - 1)

    # 4 clusters of size 768 each
    step = 3072 // 4
    c1 = centered[:, 0:step].mean(dim=1, keepdim=True)
    c2 = centered[:, step:2*step].mean(dim=1, keepdim=True)
    c3 = centered[:, 2*step:3*step].mean(dim=1, keepdim=True)
    c4 = centered[:, 3*step:].mean(dim=1, keepdim=True)
    cluster_means = torch.cat([c1, c2, c3, c4], dim=1)  # shape [N, 4]

    mean_cov = torch.matmul(cluster_means.T, cluster_means) / (N - 1)
    return cov, mean_cov, cluster_means

def plot_cov(cov, mean_cov, layer_idx, out_dir):
    # Full covariance (3072x3072) could be huge for a heatmap, but we'll do it if you want:
    plt.figure(figsize=(10,10))
    sns.heatmap(cov.cpu().numpy(), cmap="hot", annot=False)
    plt.title(f"Layer {layer_idx} - Full Cov (3072x3072)")
    plt.savefig(os.path.join(out_dir, f"covariance_matrix_{layer_idx}.png"))
    plt.close()

    # Cluster 4x4 covariance
    plt.figure(figsize=(4,4))
    sns.heatmap(mean_cov.cpu().numpy(), cmap='hot', annot=True, fmt='.3g')
    plt.title(f"Layer {layer_idx} - 4-Cluster Covariance")
    plt.savefig(os.path.join(out_dir, f"mean_covariance_matrix_{layer_idx}.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4) Activation pattern analysis
#    We'll normalize each 4D cluster vector. That can show which cluster is "dominant" per example.
# ─────────────────────────────────────────────────────────────────────────────
def plot_activation_patterns(cluster_means, layer_idx, out_dir):
    """
    cluster_means: [N, 4]
    We'll do a boxplot of normalized usage across examples.
    """
    with torch.no_grad():
        # shift if negative, then normalize
        min_val = cluster_means.min()
        offset = 0
        if min_val < 0:
            offset = -float(min_val) + 1e-5
        shifted = cluster_means + offset
        row_sums = shifted.sum(dim=1, keepdim=True) + 1e-9
        pattern = shifted / row_sums  # shape [N, 4]

    # Sample up to 1k
    N = pattern.size(0)
    sample_size = min(N, 1000)
    idxs = torch.randperm(N)[:sample_size]
    sample_data = pattern[idxs].cpu().numpy()

    import pandas as pd
    df = pd.DataFrame(sample_data, columns=[f"C{i}" for i in range(4)])
    plt.figure(figsize=(6,6))
    sns.boxplot(data=df)
    plt.title(f"Layer {layer_idx} - Normalized Activation Patterns")
    plt.ylabel("Normalized cluster level")
    plt.savefig(os.path.join(out_dir, f"activation_patterns_{layer_idx}.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5) Main analysis function
#    - Processes chunk0_*.pt files for each layer
#    - Each file is shape [batch_size, seq_len, d_model] => flatten to [N, d_model]
#    - Covariance, cluster means, MI, HSIC, activation patterns
# ─────────────────────────────────────────────────────────────────────────────
def activation_analysis():
    data_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/interp-gains-gpt2small/data_refactored"
    out_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/plots_v2"
    os.makedirs(out_dir, exist_ok=True)

    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith("chunk0")])

    num_layers = 12
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx} ...")
        layer_data_list = []

        # Collect data for this layer from all chunk0 files
        for fname in tqdm(file_list, desc=f"Layer {layer_idx} chunk0 files"):
            path = os.path.join(data_dir, fname) 
            loaded_dict = torch.load(path)  # shape: {layer_idx -> [128, 1024, 3072]}
            if layer_idx in loaded_dict:
                data_cpu = loaded_dict[layer_idx].cpu()
                # Flatten (batch_size, seq_len) => single dimension
                B, S, D = data_cpu.shape  # e.g. 128, 1024, 3072
                data_flat = data_cpu.view(B*S, D)  # [128*1024, 3072]
                layer_data_list.append(data_flat)
            del loaded_dict
            torch.cuda.empty_cache()

        if not layer_data_list:
            print(f"No data found for layer {layer_idx}, skipping.")
            continue

        # Concatenate layer data
        full_data_2d = torch.cat(layer_data_list, dim=0)  # [N, 3072]
        del layer_data_list

        # (A) Covariance + cluster means
        cov, mean_cov, cluster_means = compute_covariances(full_data_2d)
        plot_cov(cov, mean_cov, layer_idx, out_dir)

        # (B) Pairwise MI among the 4 cluster vectors
        mi_matrix = np.zeros((4,4), dtype=np.float32)
        for i in range(4):
            for j in range(i+1, 4):
                mi_val = mutual_information(cluster_means[:, i], cluster_means[:, j], bins=30)
                mi_matrix[i, j] = mi_val
                mi_matrix[j, i] = mi_val
        plt.figure(figsize=(4,4))
        sns.heatmap(mi_matrix, annot=True, cmap="Blues", fmt=".3f")
        plt.title(f"Layer {layer_idx} - MI among 4 clusters")
        plt.savefig(os.path.join(out_dir, f"mi_matrix_{layer_idx}.png"))
        plt.close()

        # (C) Pairwise HSIC among the 4 cluster vectors
        hsic_mat = np.zeros((4,4), dtype=np.float32)
        for i in range(4):
            for j in range(i+1, 4):
                h_val = hsic(cluster_means[:, i], cluster_means[:, j], sigma=1.0)
                hsic_mat[i, j] = h_val
                hsic_mat[j, i] = h_val
        plt.figure(figsize=(4,4))
        sns.heatmap(hsic_mat, annot=True, cmap="Greens", fmt=".3f")
        plt.title(f"Layer {layer_idx} - HSIC among 4 clusters")
        plt.savefig(os.path.join(out_dir, f"hsic_matrix_{layer_idx}.png"))
        plt.close()

        # (D) Activation pattern analysis
        plot_activation_patterns(cluster_means, layer_idx, out_dir)

        # Free memory
        del full_data_2d
        del cluster_means
        torch.cuda.empty_cache()

    print("All layers processed successfully.")


def main():
    activation_analysis()

if __name__ == "__main__":
    main()