#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:50:00
#SBATCH --partition=single
#SBATCH --job-name=train_run
#SBATCH --output=logs/interp-ravel-%j.out


source /data/joan_velja/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate modularity

cd /data/joan_velja/nn-modularity


entities=("city-country" "city-continent" "city-language" "object-size")

# layers=(2 5 6 7 10)
layers=(0 1 2 3 4 5)

# models=("gpt2" "pythia70m" "pythia1.4b")
# models=("gpt2")
models=("pythia70m")

type_intervention=("type1" "type2")

model_types=("model" "nmodel")

for model in "${models[@]}"; do
    for layer in "${layers[@]}"; do
        for entity in "${entities[@]}"; do
            for type in "${type_intervention[@]}"; do
                for modeltype in "${model_types[@]}"; do
                echo "Running module analysis for $model for layer $layer $type and entity $entity and model type $modeltype"
                python interpretability/ravel_analysis/ravel_module_analysis.py --device "cuda" --num_layer $layer --type_of_intervention $type --model "$model" --entity "$entity" --modeltype "$modeltype"
                done
            done
        done
    done
done
