from __init__ import *

from transformer_lens.evals import make_wiki_data_loader
from transformer_lens.HookedTransformer import HookedTransformer

try:

    with open("interp-gains-gpt2small/secret.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    access_token = config['access_token']

except:
    
    print(f"Please provide a valid access token here.")
    # access_token = None    

model = HookedTransformer.from_pretrained('gpt2-small')
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', token = access_token)

data = make_wiki_data_loader(tokenizer, batch_size=8)

for sample in data:
    print(tokenizer.decode(sample['tokens'][0], skip_special_tokens=True))
    break