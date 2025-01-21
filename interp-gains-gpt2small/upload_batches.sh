#!/bin/bash

REPO_ID="modularity_activations_gpt2"
BASE_DIR="/data/joan_velja/nn-modularity/interp-gains-gpt2small/interp-gains-gpt2small/data_refactored"

echo "Man, I am starting to upload your files"

# Loop through batches 1-8
for batch in {0..1}; do
    # Find and upload all chunk files for current batch
    find "$BASE_DIR" -name "chunk*_batch${batch}.pt" -type f | while read -r file; do
        echo "Uploading: $file"
        if huggingface-cli upload "$REPO_ID" "$file"; then
            echo "Successfully uploaded: $file"
        else
            echo "Failed to upload: $file"
        fi
    done
done