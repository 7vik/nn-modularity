from __init__ import *


def config_(args):
    model =  GPT2Model.from_pretrained('gpt2')
    
    print(model)
    
    model_nmod = HookedTransformer.from_pretrained('gpt2-small')
    model_mod = HookedTransformer.from_pretrained('gpt2-small')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device(args.device)

    model_nmod.load_state_dict(torch.load('interp-gains-gpt2small/data/models/wiki_non_modular_mlp_in_out.pt', map_location=device))
    model_mod.load_state_dict(torch.load('interp-gains-gpt2small/data/models/wiki_fully_modular_mlp_in_out.pt', map_location=device))

    logging.basicConfig(filename = 'interp-gains-gpt2small/logs/model_analysis.log',
                        level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    model_nmod.to(device)
    model_mod.to(device)
    logging.info(model_nmod)
    
    return model_nmod, model_mod, tokenizer, device



def intervention(args, index):
    '''
    We will be limiting our intervention on the modules for just 3 layers.
    Based on these intervention we will be building our data.
    
    It will be computed in two formats:
    
    1. Switch off 1 module M, i.e. keep 3 modules on 
    2. Switch off 3 modules, and keep 1 module M on
    
    '''
    
    model_nmod, model_mod, tokenizer, device = config_(args)
    
    def hook_fn(module, input, output):
        mod_output = output.clone()
        # mod_output[:, index[0]:index[1], :] = 0
        mod_output[:, :, :] = 0
        output = mod_output
        return output

    hook_ = model_mod.blocks[args.num_layer].mlp.hook_pre.register_forward_hook(hook_fn)
    
    
    # hook(index)
    
    all_loss = []
    
    for idx, data_ in tqdm(enumerate(make_wiki_data_loader(tokenizer, batch_size=args.batch_size))):
        data = data_['tokens'].to(device)
        loss = model_mod(data, return_type = "loss")
        all_loss.append(loss.item())
        if idx == 100:
            break
        
    hook_.remove()
    
    return all_loss
    

def visualize(dictionary):
    plt.figure(figsize=(10, 5))
    plt.bar(dictionary.keys(), dictionary.values())
    plt.xlabel('Layer Index')
    plt.ylabel('Loss')
    plt.show()
    plt.close()


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=3)
    
    args = parser.parse_args()
    
    index1 = [0, 1024//4]
    index2 = [1024//4, 1024//2]
    index3 = [1024//2, (1024//4)*3]
    index4 = [(1024//4)*3, 1024]
    
    layer_wise_loss_dict = {}
    
    for layer_idx in tqdm(range(12)):
        args.num_layer = layer_idx
        all_loss = intervention(args, index1)
        layer_wise_loss_dict[layer_idx] = np.mean(np.array(all_loss))
    
    visualize(layer_wise_loss_dict)
    


if __name__ == '__main__':
    main()