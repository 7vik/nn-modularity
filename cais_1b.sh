#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=single
#SBATCH --job-name=train_run
#SBATCH --output=logs//model_training-pythia70m%j.out


source /data/joan_velja/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate modularity

cd /data/joan_velja/nn-modularity/language-models/pythia-finetune

srun python -u pythia.py --model_name EleutherAI/pythia-1.4b --lr 5e-4