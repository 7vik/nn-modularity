entities=("city-country" "city-continent" "city-language" "object-size")

layers=(2 5 6 7 10)
# layers=(0 1 2 3 4 5)

models=("gpt2" "pythia70m" "pythia1.4b")

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

# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type2" --model "gpt2" --entity "city-country" --modeltype "model"
# Running module analysis for gpt2 for layer 2 type2 and entity city-country and model type model




# echo "Running module analysis for Pythia70m for layer 2 type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type1" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 5 type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 5 --type_of_intervention "type1" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 6 type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 6 --type_of_intervention "type1" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 7 type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 7 --type_of_intervention "type1" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 10 type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 10 --type_of_intervention "type1" --model "pythia70m"


# echo "Running module analysis for Pythia70m for layer 2 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type2" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 5 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 5 --type_of_intervention "type2" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 6 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 6 --type_of_intervention "type2" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 7 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 7 --type_of_intervention "type2" --model "pythia70m"

# echo "Running module analysis for Pythia70m for layer 10 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 10 --type_of_intervention "type2" --model "pythia70m"






# echo "Running module analysis for Pythia1.4m for layer 2 for type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type1" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 5 for type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 5 --type_of_intervention "type1" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 6 for type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 6 --type_of_intervention "type1" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 7 for type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 7 --type_of_intervention "type1" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 10 for type1"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 10 --type_of_intervention "type1" --model "pythia1.4b"


# echo "Running module analysis for Pythia1.4m for layer 2 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type2" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 5 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 5 --type_of_intervention "type2" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 6 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 6 --type_of_intervention "type2" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 7 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 7 --type_of_intervention "type2" --model "pythia1.4b"

# echo "Running module analysis for Pythia1.4m for layer 10 for type2"
# python interpretability/ravel_analysis/ravel_module_analysis.py --device "mps" --num_layer 10 --type_of_intervention "type2" --model "pythia1.4b"

