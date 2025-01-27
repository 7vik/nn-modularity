import os
import platform
import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans


def clusterability(
    matrix: torch.Tensor,
    cluster_U_indices: dict[int, list[Any]] | None,
    cluster_V_indices: dict[int, list[Any]] | None,
    num_clusters: int,
):
    """
    Compute the goodness of the clustering based on the intra-cluster out-degree

    Args:
        matrix: the matrix to compute the goodness of the clustering
        cluster_U_indices: a dictionary of cluster indices for U
        cluster_V_indices: a dictionary of cluster indices for V
        num_clusters: the number of clusters

    Returns:
        goodness: the goodness of the clustering
    """

    A = matrix**2
    mask = torch.zeros_like(A, dtype=torch.bool)

    if cluster_U_indices is None or cluster_V_indices is None:
        cluster_size = (A.shape[0] // num_clusters, A.shape[1] // num_clusters)
        cluster_U_indices = {
            i: list(range(i * cluster_size[0], (i + 1) * cluster_size[0]))
            for i in range(num_clusters)
        }
        cluster_V_indices = {
            i: list(range(i * cluster_size[1], (i + 1) * cluster_size[1]))
            for i in range(num_clusters)
        }

    for cluster_idx in range(num_clusters):
        u_indices = torch.tensor(cluster_U_indices[cluster_idx], dtype=torch.long)
        v_indices = torch.tensor(cluster_V_indices[cluster_idx], dtype=torch.long)
        mask[u_indices.unsqueeze(1), v_indices] = True

    intra_cluster_out_sum = torch.sum(A[mask])
    total_out_sum = torch.sum(A)

    return intra_cluster_out_sum / total_out_sum


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


def is_mps_available() -> bool:
    """
    Safely check if MPS (Metal Performance Shaders) is available.
    """
    if platform.system() != "Darwin":  # MPS is only available on macOS
        return False

    # Check if the current PyTorch version has MPS support
    if not hasattr(torch, "backends") or not hasattr(torch.backends, "mps"):
        return False

    return torch.backends.mps.is_available()


def set_all_seeds(
    seed: int, deterministic: bool = True, warn_only: bool = False
) -> None:
    """
    Set all seeds and deterministic flags for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators
        deterministic (bool): Whether to enforce deterministic behavior
        warn_only (bool): If True, warning instead of error when deterministic
                        operations aren't supported
    """
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)

    # Handle CUDA devices
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU
        except Exception as e:
            print(f"Warning: Could not set CUDA seeds: {str(e)}")

    # Handle MPS device (Apple Silicon)
    if is_mps_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception as e:
            print(f"Warning: Could not set MPS seed: {str(e)}")

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        try:
            # CUDA deterministic behavior
            if torch.cuda.is_available():
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # Set deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

        except Exception as e:
            msg = f"Warning: Could not enable deterministic mode. Error: {str(e)}"
            if not warn_only:
                raise RuntimeError(msg)
            print(msg)


def get_device() -> torch.device:
    """
    Get the most appropriate PyTorch device available.

    Returns:
        torch.device: The preferred device (CUDA > MPS > CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def autotune_batch_size(
    model, tokenizer, block_size=512, max_batch=32, safety_margin=0.85
) -> int:
    """
    Autotune the maximum batch size for a given model and tokenizer.

    Args:
        model: The PyTorch model to autotune
        tokenizer: The tokenizer for the model
        block_size: The maximum sequence length
        max_batch: The maximum batch size to search for
        safety_margin: The fraction of GPU memory to utilize

    Returns:
        int: The maximum batch size that fits within the safety margin
    """

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def create_dummy_batch(batch_size):
        return {
            "input_ids": torch.full(
                (batch_size, block_size),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            ),
            "attention_mask": torch.ones(
                (batch_size, block_size), dtype=torch.long, device=device
            ),
        }

    # Find maximum possible batch size through exponential search
    low, high = 1, max_batch
    best_size = 1

    while low <= high:
        mid = (low + high) // 2
        torch.cuda.empty_cache()

        try:
            dummy = create_dummy_batch(mid)
            outputs = model(**dummy, labels=dummy["input_ids"])
            loss = outputs.loss
            loss.backward()
            model.zero_grad()

            # Check memory utilization with safety margin
            total_mem = torch.cuda.get_device_properties(device).total_memory
            used_mem = torch.cuda.max_memory_allocated(device)

            if used_mem < total_mem * safety_margin:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                high = mid - 1
            else:
                raise e

    return max(1, best_size)
