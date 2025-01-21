import os
import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

matplotlib.rcParams.update({"font.size": 12})

###############################################################################
# Incremental Covariance (no NxN or [N, D] storage)
###############################################################################
class IncrementalCovariance:
    def __init__(self, d_model: int):
        self.d_model = d_model
        # Store sums in float64 for numerical stability
        self.M = torch.zeros(d_model, dtype=torch.float64)           # sum of x_k
        self.Q = torch.zeros(d_model, d_model, dtype=torch.float64)  # sum of x_k x_k^T
        self.count = 0

    def update(self, x: torch.Tensor):
        """
        x: [B, D] chunk
        """
        if x.dtype != torch.float64:
            x = x.double()

        B = x.size(0)
        sum_x = x.sum(dim=0)           # [D]
        sum_x2 = x.t().mm(x)          # [D, D]

        self.M += sum_x
        self.Q += sum_x2
        self.count += B

    def finalize(self):
        """
        Returns sample covariance [D, D] as float32
        """
        if self.count < 2:
            raise ValueError("Not enough data to compute covariance.")
        M_reshaped = self.M.unsqueeze(1)   # [D,1]
        outer_MM = (M_reshaped @ M_reshaped.t()) / self.count   # [D, D]

        cov = (self.Q - outer_MM) / (self.count - 1)
        return cov.float()


###############################################################################
# Reservoir Sampling (randomly pick up to max_size rows)
###############################################################################
class ReservoirSampler:
    """
    Maintains a random sample of rows up to max_size.
    Simple reservoir sampling approach.
    """

    def __init__(self, max_size: int, d_model: int):
        self.max_size = max_size
        self.d_model = d_model
        self.samples = []
        self.count = 0

    def update(self, x: torch.Tensor):
        """
        x: [B, D]
        We'll store up to max_size rows in a random fashion.
        """
        B = x.size(0)
        for i in range(B):
            self.count += 1
            if len(self.samples) < self.max_size:
                self.samples.append(x[i].clone())
            else:
                # Reservoir sampling
                r = random.randint(0, self.count - 1)
                if r < self.max_size:
                    self.samples[r] = x[i].clone()

    def finalize(self):
        """
        Return a single tensor [K, D] with the sub-sample collected.
        """
        if len(self.samples) == 0:
            return None
        return torch.stack(self.samples, dim=0)


###############################################################################
# Mutual Information (1D => 1D)
###############################################################################
def mutual_information(x: np.ndarray, y: np.ndarray, bins=30) -> float:
    joint_hist2d, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
    joint_prob = joint_hist2d / joint_hist2d.sum()
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)

    mi_val = 0.0
    for i in range(len(x_prob)):
        for j in range(len(y_prob)):
            if joint_prob[i, j] > 0:
                mi_val += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_prob[i]*y_prob[j]))
    return float(mi_val)


###############################################################################
# HSIC (Hilbert-Schmidt Indep. Criterion) for 1D => 1D using RBF
###############################################################################
def rbf_kernel(array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if array.ndim == 1:
        array = array[:, None]
    XX = np.sum(array**2, axis=1, keepdims=True)
    D = XX - 2*np.dot(array, array.T) + XX.T
    return np.exp(-D / (2*sigma*sigma))

def hsic_1d(x: np.ndarray, y: np.ndarray, sigma=1.0) -> float:
    x = x[:, None]
    y = y[:, None]
    Kx = rbf_kernel(x, sigma)
    Ky = rbf_kernel(y, sigma)

    n = Kx.shape[0]
    H = np.eye(n) - (1.0/n)*np.ones((n,n))
    hsic_val = np.trace(Kx.dot(H).dot(Ky).dot(H)) / ((n - 1)**2)
    return float(hsic_val)


###############################################################################
# Utility: Plot a 4x4 correlation or covariance matrix
###############################################################################
def plot_matrix_4x4(mat: np.ndarray, title: str, out_path: str, center_zero=False):
    """
    mat: shape [4,4]
    If center_zero=True, we use a diverging palette good for negative vs. positive.
    """
    plt.figure(figsize=(5,4.5))
    if center_zero:
        cmap = "RdBu_r"
        vmin, vmax = -1, 1
    else:
        cmap = "Reds"
        vmin, vmax = None, None
    
    ax = sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, square=True)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


###############################################################################
# Main Analysis: 
#  - Incremental Cov
#  - Subsample for cluster means, MI, HSIC
#  - Plot both Cov and Correlation in the 4x4 cluster matrix
###############################################################################
def activation_analysis():
    data_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/interp-gains-gpt2small/data_refactored"
    out_dir = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/plots-v2"
    os.makedirs(out_dir, exist_ok=True)

    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith("chunk0")])
    num_layers = 12
    d_model = 3072
    sample_size = 5000
    
    for layer_idx in range(num_layers):
        print(f"\n== Processing layer {layer_idx} ==")

        # A) Build incremental covariance + reservoir sampler
        inc_cov = IncrementalCovariance(d_model)
        sampler = ReservoirSampler(sample_size, d_model)

        # B) Load data chunk by chunk
        for fname in tqdm(file_list, desc=f"Layer {layer_idx}"):
            data_path = os.path.join(data_dir, fname)
            data_dict = torch.load(data_path)  # {layer_idx -> [B,S,D]}
            if layer_idx not in data_dict:
                continue
            data_cpu = data_dict[layer_idx].cpu()
            B, S, D = data_cpu.shape  # e.g. [128, 1024, 3072]
            # Flatten => [B*S, D]
            flattened = data_cpu.view(B*S, D)

            # Update incremental cov & reservoir
            inc_cov.update(flattened)
            sampler.update(flattened)

            del data_dict, data_cpu, flattened

        if inc_cov.count < 2:
            print(f"No data for layer {layer_idx}; skipping.")
            continue

        # C) Finalize full covariance
        full_cov = inc_cov.finalize()  # shape [3072, 3072]
        # We store it to disk, but won't try to plot a huge heatmap
        torch.save(full_cov, os.path.join(out_dir, f"layer{layer_idx}_cov.pt"))
        print(f"Saved large covariance for layer {layer_idx} to disk (shape={full_cov.shape}).")

        # D) Subsample data for cluster analysis
        sample_data = sampler.finalize()
        if sample_data is None or sample_data.size(0) < 2:
            print(f"No sub-sampled data for layer {layer_idx}; skipping analysis.")
            continue

        # E) 4 cluster means from sub-sample
        #   1) Center
        mean_ = sample_data.mean(dim=0, keepdim=True)
        centered = sample_data - mean_
        #   2) 4 groups
        step = d_model // 4
        c1 = centered[:, :step].mean(dim=1, keepdim=True)
        c2 = centered[:, step:2*step].mean(dim=1, keepdim=True)
        c3 = centered[:, 2*step:3*step].mean(dim=1, keepdim=True)
        c4 = centered[:, 3*step:].mean(dim=1, keepdim=True)
        cluster_means = torch.cat([c1, c2, c3, c4], dim=1)  # [K,4]

        # F) 4x4 Covariance among cluster means
        K_ = cluster_means.size(0)
        c_cov = (cluster_means.t() @ cluster_means) / (K_ - 1)  # shape [4,4]
        c_cov_np = c_cov.cpu().numpy()

        # Also 4x4 Correlation (normalize the diagonal)
        diag = np.sqrt(np.diag(c_cov_np))  # [4]
        outer_diag = np.outer(diag, diag)
        eps = 1e-12
        cor_np = c_cov_np / (outer_diag + eps)  # shape [4,4]

        # Plot them
        plot_matrix_4x4(
            c_cov_np, 
            title=f"Layer {layer_idx} – 4x4 Cov (clusters, sub-sample)", 
            out_path=os.path.join(out_dir, f"mean_covariance_matrix_{layer_idx}.png"),
            center_zero=False  # Cov can be >0 mostly
        )
        plot_matrix_4x4(
            cor_np, 
            title=f"Layer {layer_idx} – 4x4 Corr (clusters, sub-sample)", 
            out_path=os.path.join(out_dir, f"mean_correlation_matrix_{layer_idx}.png"),
            center_zero=True    # correlation can be negative or positive
        )

        # G) Pairwise MI & HSIC
        cluster_np = cluster_means.cpu().numpy()  # shape [K,4]
        mi_mat = np.zeros((4,4), dtype=np.float32)
        hsic_mat = np.zeros((4,4), dtype=np.float32)
        for i in range(4):
            for j in range(i+1, 4):
                x_ = cluster_np[:, i]
                y_ = cluster_np[:, j]
                mi_val = mutual_information(x_, y_, bins=30)
                mi_mat[i,j] = mi_val
                mi_mat[j,i] = mi_val

                h_val = hsic_1d(x_, y_, sigma=1.0)
                hsic_mat[i,j] = h_val
                hsic_mat[j,i] = h_val

        # Plot MI & HSIC (4x4)
        plt.figure(figsize=(5,4.5))
        ax = sns.heatmap(mi_mat, annot=True, fmt=".2f", cmap="Purples", square=True)
        ax.set_title(f"Layer {layer_idx} – MI among 4 clusters", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mi_matrix_{layer_idx}.png"))
        plt.close()

        plt.figure(figsize=(5,4.5))
        ax = sns.heatmap(hsic_mat, annot=True, fmt=".3f", cmap="Greens", square=True)
        ax.set_title(f"Layer {layer_idx} – HSIC among 4 clusters", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hsic_matrix_{layer_idx}.png"))
        plt.close()

        # H) Activation pattern analysis (Boxplot)
        min_val = cluster_means.min().item()
        offset = 0.0
        if min_val < 0:
            offset = -min_val + 1e-5
        shifted = cluster_means + offset
        sums = shifted.sum(dim=1, keepdim=True) + 1e-9
        pattern = shifted / sums
        K_ = pattern.size(0)
        n_sub = min(K_, 1000)
        idxs = torch.randperm(K_)[:n_sub]
        sample_pat = pattern[idxs].cpu().numpy()

        import pandas as pd
        df = pd.DataFrame(sample_pat, columns=[f"Cluster{i}" for i in range(4)])
        plt.figure(figsize=(6,5))
        ax = sns.boxplot(data=df, palette="Set2")
        ax.set_title(f"Layer {layer_idx} – Normalized Activation Patterns", fontsize=14)
        ax.set_ylabel("Relative cluster activation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"activation_patterns_{layer_idx}.png"))
        plt.close()

        print(f"Done layer {layer_idx}")

    print("\nAll layers processed successfully.")


def main():
    activation_analysis()


if __name__ == "__main__":
    main()