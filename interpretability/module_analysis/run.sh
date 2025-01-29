layers=(2 5 6 7 10)

# models=("pythia70m" "pythia1.4b")
models=("pythia70m")

type_intervention=("type1" "type2")

model_types=("model" "nmodel")
# model_types=("model")

for model in "${models[@]}"; do
    for layer in "${layers[@]}"; do
        for type in "${type_intervention[@]}"; do
            for modeltype in "${model_types[@]}"; do
            echo "Running module analysis for $model for layer $layer $type and entity $entity and model type $modeltype"
            python interpretability/module_analysis/module_analysis.py --device "cuda" --num_layer $layer --type_of_intervention $type --model "$model" --entity "$entity" --modeltype "$modeltype"
                # done
            done
        done
    done
done
