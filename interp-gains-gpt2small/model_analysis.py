from __init__ import *

from transformer_lens.evals import make_wiki_data_loader
from transformer_lens.HookedTransformer import HookedTransformer

model_nmod = HookedTransformer.from_pretrained('gpt2-small')
model_mod = HookedTransformer.from_pretrained('gpt2-small')
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

device = torch.device("cpu")

model_nmod.load_state_dict(torch.load('interp-gains-gpt2small/data/wiki_non_modular_mlp_in_out.pt', map_location=device))
model_mod.load_state_dict(torch.load('interp-gains-gpt2small/data/wiki_fully_modular_mlp_in_out.pt', map_location=device))

model_nmod.to(device)
# model_mod.to(device)
print(model_nmod)



class hook:
    
    def __init__(self, model):
        self.model = model
        self.hook = self.model.blocks[0].mlp.hook_pre.register_forward_hook(self.hook_fn)
        self.outputs = []
    
    def hook_fn(self, module, input, output):
        self.outputs.append(output)
        return output

    def forward(self):
        return self.outputs
    
    
    

for data_ in make_wiki_data_loader(tokenizer, batch_size=8):
    data = data_['tokens'].to(device)
    # logits, clean_cache = model_nmod.run_with_cache(data)
    activation = hook(model_nmod)
    logits = model_nmod(data)
    output = activation.forward()
    print(output[0].size())
    break


