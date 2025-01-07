#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=single
#SBATCH --job-name=pickle_layers_gpt2small_refactored
#SBATCH --output=logs/refactored/model_analysis-%j.out


source /data/joan_velja/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate modularity

cd /data/joan_velja/nn-modularity/interp-gains-gpt2small

srun python -u model_analysis_refactored.py --device "cuda" --batch_size 128