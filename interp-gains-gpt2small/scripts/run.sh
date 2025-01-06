chmod +x interp-gains-gpt2small/scripts/setup.sh

./interp-gains-gpt2small/scripts/setup.sh

python interp-gains-gpt2small/model_analysis.py --device "cuda" --batch_size 128
