#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:50:00
#SBATCH --partition=single
#SBATCH --job-name=train_run
#SBATCH --output=logs/1.4b-NonModular-1e-4-mix-%j.out


source /data/joan_velja/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate modularity

cd /data/joan_velja/nn-modularity/language-models/pythia-finetune

srun python -u pythia.py --model_name EleutherAI/pythia-1.4b --mix_data --lr 1e-4