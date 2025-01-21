# full layer covariance plots

from pathlib import Path

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import torch


# Helper functions for analysis
def statistical_tests(cov_matrix):
    """Perform statistical tests on the covariance matrix"""
    cluster_size = 3072 // 4
    within_corrs = []
    between_corrs = []

    tri_idx = torch.triu_indices(cluster_size, cluster_size, offset=1)
    # Gather correlations
    for i in range(4):
        # Within cluster correlations
        cluster = cov_matrix[
            i * cluster_size : (i + 1) * cluster_size,
            i * cluster_size : (i + 1) * cluster_size,
        ]
        within_vals = cluster[tri_idx[0], tri_idx[1]]
        within_corrs.extend(within_vals.tolist())

        # Between cluster correlations
        for j in range(i + 1, 4):
            block = cov_matrix[
                i * cluster_size : (i + 1) * cluster_size,
                j * cluster_size : (j + 1) * cluster_size,
            ]
            between_corrs.extend(block.flatten().tolist())

    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(within_corrs, between_corrs)
    return t_stat, p_value


def perturbation_analysis(cov_matrix, n_permutations=1000):
    """Test if observed modularity is significant compared to random clustering"""
    true_score = compute_cluster_metrics(cov_matrix)["modularity"]
    random_scores = []

    for _ in range(n_permutations):
        # Randomly permute rows and columns
        perm = torch.randperm(cov_matrix.size(0))
        permuted = cov_matrix[perm][:, perm]
        random_scores.append(compute_cluster_metrics(permuted)["modularity"])

    p_value = sum(1 for s in random_scores if s >= true_score) / len(random_scores)
    return p_value


def compute_cluster_metrics(cov_matrix):
    """Compute modularity metrics for the covariance matrix"""
    # Split into 4 equal clusters of size 768
    cluster_size = 3072 // 4
    clusters = []
    for i in range(4):
        start_idx = i * cluster_size
        end_idx = (i + 1) * cluster_size
        clusters.append(cov_matrix[start_idx:end_idx, start_idx:end_idx])

    # Calculate metrics
    # within_cluster_corr = np.mean([np.mean(np.abs(c)) for c in clusters])
    withins = []
    for c in clusters:
        withins.append(torch.mean(torch.abs(c)).item())
    within_cluster_corr = sum(withins) / len(withins)

    betweens = []
    for i in range(4):
        for j in range(i + 1, 4):
            block = cov_matrix[
                i * cluster_size : (i + 1) * cluster_size,
                j * cluster_size : (j + 1) * cluster_size,
            ]
            betweens.append(torch.mean(torch.abs(block)).item())
    between_cluster_corr = sum(betweens) / len(betweens)

    modularity_score = within_cluster_corr - between_cluster_corr
    return {
        "within_cluster": within_cluster_corr,
        "between_cluster": between_cluster_corr,
        "modularity": modularity_score,
    }


def plot_covariance_heatmap(cov_matrix, layer_idx, save_path):
    """Plot covariance matrix heatmap and cluster-wise summary"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Full matrix plot with reduced resolution
    # downsampled = cov_matrix[::4, ::4]  # Downsample for visualization
    downsampled = cov_matrix[::4, ::4].cpu().numpy()
    sns.heatmap(
        downsampled,
        cmap="RdBu_r",
        center=0,
        ax=ax1,
        xticklabels=False,
        yticklabels=False,
    )
    ax1.set_title(f"Layer {layer_idx} Covariance (Downsampled)")

    # Cluster-level summary
    cluster_size = 3072 // 4
    # cluster_means = np.zeros((4, 4))
    cluster_means = torch.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            block = cov_matrix[
                i * cluster_size : (i + 1) * cluster_size,
                j * cluster_size : (j + 1) * cluster_size,
            ]
            cluster_means[i, j] = torch.mean(block)

    # Normalize for visualization
    cluster_means = cluster_means.cpu().numpy()
    cluster_means -= cluster_means.mean()
    cluster_means /= cluster_means.std()

    sns.heatmap(cluster_means, cmap="RdBu_r", center=0, ax=ax2, annot=True, fmt=".2f")
    ax2.set_title(f"Layer {layer_idx} Cluster-wise Mean Covariance")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_and_analyze_covariance(
    plot_dir="/data/joan_velja/nn-modularity/interp-gains-gpt2small/plots-v2/layer_cov_plots",
):
    """Load covariance matrices and analyze them"""
    # Main analysis loop
    results = {}
    plot_path = Path(plot_dir)
    # make sure the plot path exists
    plot_path.mkdir(parents=True, exist_ok=True)
    data_path = "/data/joan_velja/nn-modularity/interp-gains-gpt2small/plots-v2"  # data stored as .pt files
    # Naming of files: layer{layer_idx}_cov.pt

    for layer_file in sorted(Path(data_path).glob("layer*_cov.pt")):
        print(f"Processing {layer_file}")
        layer_idx = int(layer_file.stem.split("_")[0][-1])

        cov_matrix = torch.load(layer_file)

        # Compute metrics
        metrics = compute_cluster_metrics(cov_matrix)
        results[layer_idx] = metrics

        # Plot heatmap
        save_path = plot_path / f"layer_{layer_idx}_analysis.png"
        plot_covariance_heatmap(cov_matrix, layer_idx, save_path)

        t_stat, p_value = statistical_tests(cov_matrix)
        perm_p_value = perturbation_analysis(cov_matrix)
        results[layer_idx].update(
            {
                "t_statistic": t_stat,
                "p_value": p_value,
                "permutation_p_value": perm_p_value,
            }
        )

    # Plot metrics across layers
    plt.figure(figsize=(10, 6))
    layers = sorted(results.keys())
    modularity_scores = [results[l]["modularity"] for l in layers]
    plt.plot(layers, modularity_scores, "o-")
    plt.xlabel("Layer")
    plt.ylabel("Modularity Score")
    plt.title("Modularity Score Across Layers")
    plt.savefig(plot_path / "modularity_across_layers.png")
    plt.close()

    return results


if __name__ == "__main__":
    results = load_and_analyze_covariance()
    print(results)
