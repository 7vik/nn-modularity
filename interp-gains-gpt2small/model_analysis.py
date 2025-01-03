from __init__ import *

model_nmod = HookedTransformer.from_pretrained('gpt2-small')
model_mod = HookedTransformer.from_pretrained('gpt2-small')
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

device = torch.device("cpu")

model_nmod.load_state_dict(torch.load('interp-gains-gpt2small/data/wiki_non_modular_mlp_in_out.pt', map_location=device))
model_mod.load_state_dict(torch.load('interp-gains-gpt2small/data/wiki_fully_modular_mlp_in_out.pt', map_location=device))

logging.basicConfig(filename = 'interp-gains-gpt2small/logs/model_analysis.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

model_nmod.to(device)
# model_mod.to(device)
# logging.info(model_nmod)



class hook:
    
    def __init__(self, model, layer_idx):
        self.model = model
        self.hook = self.model.blocks[layer_idx].mlp.hook_pre.register_forward_hook(self.hook_fn)
        self.output = None
    
    def hook_fn(self, module, input, output):
        self.output = output
        return output

    def forward(self):
        return self.output
    
    
    
layer_activation = {idx: [] for idx in range(12)}


for data_ in make_wiki_data_loader(tokenizer, batch_size=1):
    for layer_idx in tqdm(range(12)):
        data = data_['tokens'].to(device)
        # logits, clean_cache = model_nmod.run_with_cache(data)
        activation = hook(model_nmod, layer_idx)
        logits = model_nmod(data)
        output = activation.forward()
        layer_activation[layer_idx].append(output)
        logging.info(f"Latest added activation shape: {layer_activation[layer_idx][-1].size()}")
        logging.info(f"Layer {layer_idx} activation length: {len(layer_activation[layer_idx])}")


os.makdirs("interp-gains-gpt2small/data", exist_ok=True)
with open('interp-gains-gpt2small/data/layer_activation_nmod.pkl', 'wb') as f:
    pkl.dump(layer_activation, f)
    
    
logging.info("Done\n\n\n")


