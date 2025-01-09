#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=single
#SBATCH --job-name=activation_analysis
#SBATCH --output=logs/refactored/activation_anaylsis/model_analysis_v2-%j.out


source /data/joan_velja/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate modularity

cd /data/joan_velja/nn-modularity/interp-gains-gpt2small

srun python -u activation_v2.py