echo "Running module analysis for GPT-2 small for layer 2"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type1"

echo "Running module analysis for GPT-2 small for layer 5"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 5 --type_of_intervention "type1"

# echo "Running module analysis for GPT-2 small for layer 6"
# python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 6 --type_of_intervention "type1"

echo "Running module analysis for GPT-2 small for layer 7"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 7 --type_of_intervention "type1"

echo "Running module analysis for GPT-2 small for layer 10"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 10 --type_of_intervention "type1"

echo "Running module analysis for GPT-2 small for layer 2"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 2 --type_of_intervention "type2"

echo "Running module analysis for GPT-2 small for layer 5"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 5 --type_of_intervention "type2"

echo "Running module analysis for GPT-2 small for layer 6"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 6 --type_of_intervention "type2"

echo "Running module analysis for GPT-2 small for layer 7"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 7 --type_of_intervention "type2"

echo "Running module analysis for GPT-2 small for layer 10"
python interp-gains-gpt2small/module_analysis.py --device "mps" --num_layer 10 --type_of_intervention "type2"
