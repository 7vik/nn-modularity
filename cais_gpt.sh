#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=single
#SBATCH --job-name=train_run
#SBATCH --output=logs/gpt2-NonModular-%j.out



source /data/joan_velja/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate modularity

cd /data/joan_velja/nn-modularity/language-models/pythia-finetune

srun python pythia.py \
    --model_name gpt2 \
    --lr 5e-4 \
    --num_epochs 2 \
    --mix_data