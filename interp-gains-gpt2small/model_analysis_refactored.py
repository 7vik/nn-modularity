
from __init__ import *
from config import PATHS
import torch
import torch.nn as nn
import logging
import os
import math
from tqdm import tqdm

# For hooking multiple layers in a single pass
class MultiLayerActivationHook:
    def __init__(self, model, layer_indices):
        """
        Registers forward hooks for all specified layers in 'layer_indices'.
        """
        self.model = model
        self.layer_indices = layer_indices
        self.hooks = []
        self.outputs = {}
        
        for idx in layer_indices:
            # We'll store each layer’s activations in self.outputs[idx]
            self.outputs[idx] = None

            def _make_hook(layer_id):
                def _hook_fn(module, inp, out):
                    # Move activation to CPU right away
                    self.outputs[layer_id] = out.detach().cpu()
                    return out
                return _hook_fn

            h = self.model.blocks[idx].mlp.hook_pre.register_forward_hook(
                _make_hook(idx)
            )
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        for idx in self.outputs:
            self.outputs[idx] = None

@torch._dynamo.disable
def forward_once(model, tokens):
    """
    Disable Dynamo just for the HookedTransformer forward
    (due to the context manager inside transformer_lens).
    """
    return model(tokens)

def setup_logging():
    """Configure logging for the project"""
    PATHS["logs"].parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=str(PATHS["logs"]),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def get_activations(args):
    """
    Process GPT-2 hidden activations in "chunks" of layers, so that we
    don't have to do 12 full passes *and* we don't blow out memory by 
    storing all 12 layers' activations for all batches at once.
    """
    device = torch.device(args.device)

    # Load the model
    model_nmod = HookedTransformer.from_pretrained('gpt2-small')
    model_nmod.load_state_dict(
        torch.load(PATHS["models"]["non_modular"], map_location=device)
    )
    model_nmod.to(device)
    model_nmod.eval()

    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create data directory
    save_root = "interp-gains-gpt2small/data_refactored"
    os.makedirs(save_root, exist_ok=True)

    # Decide how many layers per pass.  If you have plenty of memory, set =12.
    # Otherwise set 3 or 4, so you do only 3-4 passes total, rather than 12.
    CHUNK_SIZE = 12

    # Partition the 12 layers into consecutive chunks, e.g. [0,1,2], [3,4,5], etc.
    all_layers = list(range(12))
    layer_chunks = [
        all_layers[i : i + CHUNK_SIZE]
        for i in range(0, len(all_layers), CHUNK_SIZE)
    ]

    # We'll iterate through the dataset once per chunk. 
    # That is much faster than the old approach of once per layer.
    for chunk_i, chunk_layers in enumerate(layer_chunks):
        logging.info(f"Processing chunk {chunk_i} with layers {chunk_layers}")

        # Prepare hooking for this chunk
        multi_hook = MultiLayerActivationHook(model_nmod, chunk_layers)

        # We accumulate results batch by batch so we’re not storing the entire dataset in RAM
        batch_loader = make_wiki_data_loader(
            tokenizer, batch_size=args.batch_size
        )

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            for batch_idx, sample in enumerate(tqdm(batch_loader)):
                data = sample["tokens"].to(device, non_blocking=True)

                # Single forward pass triggers all hooks in chunk
                _ = forward_once(model_nmod, data)

                # multi_hook.outputs now has CPU copies of each layer’s activation
                # We'll store them to disk in a single .pt file per batch
                # Data structure to save: a dict {layer_idx: activation_tensor}
                batch_acts = {
                    lidx: multi_hook.outputs[lidx]
                    for lidx in chunk_layers
                }

                save_pt = os.path.join(
                    save_root,
                    f"chunk{chunk_i}_batch{batch_idx}.pt"
                )
                torch.save(batch_acts, save_pt)

                # Clear the dictionary so we don't accumulate in RAM
                multi_hook.clear()

                # Clear GPU cache every few steps (with a big enough batch size ~128, this is never triggered)
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

        multi_hook.remove_hooks()
        logging.info(f"Done chunk {chunk_i} with layers {chunk_layers}")

    logging.info("All chunks complete.")

def main():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    setup_logging()
    get_activations(args)

if __name__ == "__main__":
    main()