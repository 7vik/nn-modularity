import pickle as pkl
import torch
import transformers
from tqdm import tqdm

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

paths = ["interp-gains-gpt2small/data/samples_type2_layer2_mod1.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer2_mod2.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer2_mod3.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer2_mod4.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer5_mod1.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer5_mod2.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer5_mod3.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer5_mod4.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer6_mod1.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer6_mod2.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer6_mod3.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer6_mod4.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer7_mod1.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer7_mod2.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer7_mod3.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer7_mod4.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer10_mod1.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer10_mod2.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer10_mod3.pkl",
        "interp-gains-gpt2small/data/samples_type2_layer10_mod4.pkl"]


for path in tqdm(paths):
    
    samples = []
    
    with open(path, "rb") as f:
        data = pkl.load(f)

    for d in data:
        samples.append(tokenizer.decode(d.detach().cpu()[0]))

    mod_path = path.split(".pkl")[0] + ".txt"
        
    with open(mod_path, "w") as f:
        for s in samples:
            f.write(s + "\n")