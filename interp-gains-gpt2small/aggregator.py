import glob
import logging
import os
import re

import torch


def layerwise_aggregation(data_dir, out_dir, max_layers=12):
    """
    1) Looks for files named like 'chunk*_batch*.pt' inside data_dir.
    2) Each file is a dict {layer_idx: activation_tensor on CPU}.
    3) Aggregates them layer-by-layer and writes out one file per layer
       (or processes them as needed).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prepare a data structure to hold the activations for each layer
    # We'll store a list of CPU tensors for each layer to keep them in memory
    # If memory is too large, you can chunk or process them on the fly.
    layerwise_acts = {layer_idx: [] for layer_idx in range(max_layers)}

    # Grab the chunk–batch files
    # Adjust pattern if your filenames differ
    file_list = sorted(glob.glob(os.path.join(data_dir, "chunk*_batch*.pt")))

    logging.info(f"Found {len(file_list)} chunk–batch files in {data_dir}")

    # Regex to parse out chunk_i and batch_j if you need them
    # or you can remain indifferent to chunk/batch indexing
    # chunk(\d+)_batch(\d+)\.pt
    chunk_re = re.compile(r"chunk(\d+)_batch(\d+)\.pt")

    for fpath in file_list:
        fname = os.path.basename(fpath)
        match = chunk_re.search(fname)
        if not match:
            continue  # Doesn't match the naming scheme

        chunk_i = int(match.group(1))
        batch_i = int(match.group(2))

        file_dict = torch.load(fpath)  # load the dict {layer_idx: activation_tensor}

        for layer_idx, acts_tensor in file_dict.items():
            layerwise_acts[layer_idx].append(acts_tensor)

        logging.info(
            f"Loaded chunk {chunk_i}, batch {batch_i}, containing layers: {list(file_dict.keys())}"
        )

    # Now layerwise_acts[layer] is a list of Tensors (one per batch).
    # You can either:
    #   1) Concatenate them along the batch dimension (if shape matches).
    #   2) Save them out as is.
    #   3) Do your own post-processing.

    # Example 1: unify them into a single big tensor for each layer (if sizes match)
    # Then write them out as layerX.pt
    for layer_idx in range(max_layers):
        # If the batch dimension matches up, you can do:
        #   layer_cat = torch.cat(layerwise_acts[layer_idx], dim=0)
        # If shapes differ, you might skip or handle differently...
        try:
            layer_cat = torch.cat(layerwise_acts[layer_idx], dim=0)
        except RuntimeError as e:
            logging.warning(
                f"Could not concatenate layer {layer_idx} (mismatched shapes?) – {e}"
            )
            # Fallback: just store a list of tensors
            layer_cat = layerwise_acts[layer_idx]

        out_file = os.path.join(out_dir, f"layer_{layer_idx}.pt")
        torch.save(layer_cat, out_file)
        logging.info(f"Wrote aggregated layer {layer_idx} to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with chunk*_batch*.pt files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write aggregated layerwise data",
    )
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    if args.logfile:
        logging.basicConfig(level=logging.INFO, filename=args.logfile, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)

    layerwise_aggregation(args.data_dir, args.out_dir, args.max_layers)