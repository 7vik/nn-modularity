from __init__ import *


def get_activations(args):
    
    model =  GPT2Model.from_pretrained('gpt2')
    
    print(model)
    
    model_nmod = HookedTransformer.from_pretrained('gpt2-small')
    model_mod = HookedTransformer.from_pretrained('gpt2-small')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device(args.device)

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
            self.output = output.detach().cpu()
            return output

        def forward(self):
            return self.output
        
        
        
    layer_activation = {idx: [] for idx in range(12)}

    i = 0 

    for layer_idx in range(12):
        for data_ in tqdm(make_wiki_data_loader(tokenizer, batch_size=args.batch_size)):
            data = data_['tokens'].to(device)
            # logits, clean_cache = model_nmod.run_with_cache(data)
            activation = hook(model_nmod, layer_idx)
            logits = model_nmod(data)
            output = activation.forward()
            activation.hook.remove()
            layer_activation[layer_idx].append(output)
            logging.info(f"Latest added activation shape: {layer_activation[layer_idx][-1].size()}")
            logging.info(f"Layer {layer_idx} activation length: {len(layer_activation[layer_idx])}")

            os.makedirs("interp-gains-gpt2small/data", exist_ok=True)
            with open(f'interp-gains-gpt2small/data/layer_activation_nmod_{layer_idx}.pkl', 'wb') as f:
                pkl.dump(layer_activation, f)
        
        
    logging.info("Done\n\n\n")
    
    
    
def activation_analysis(args):
    

    def covariance_matrix(batch_):
        values = torch.norm(batch_, dim = -1)
        mean = values.mean(dim = 0)
        centered = values - mean
        
        assert centered.size(1) == 1024
        # cluster1 = torch.mean(centered[:, :centered.size(1)//4], dim = 1).reshape(-1, 1)
        # cluster2 = torch.mean(centered[:, centered.size(1)//4:centered.size(1)//2], dim = 1).reshape(-1, 1)
        # cluster3 = torch.mean(centered[:, centered.size(1)//2:centered.size(1)*3//4], dim = 1).reshape(-1, 1)
        # cluster4 = torch.mean(centered[:, centered.size(1)*3//4:], dim = 1).reshape(-1, 1)
        
        cluster1 = torch.mean(values[:, :values.size(1)//4], dim = 1).reshape(-1, 1)
        cluster2 = torch.mean(values[:, values.size(1)//4:centered.size(1)//2], dim = 1).reshape(-1, 1)
        cluster3 = torch.mean(values[:, values.size(1)//2:centered.size(1)*3//4], dim = 1).reshape(-1, 1)
        cluster4 = torch.mean(values[:, values.size(1)*3//4:], dim = 1).reshape(-1, 1)
        
        all_cluster_ = torch.cat([cluster1, cluster2, cluster3, cluster4], dim = 1)
        assert all_cluster_.size() == (centered.size(0), 4)
        
        all_cluster = all_cluster_ - all_cluster_.mean(dim = 0)
        
        cov = torch.matmul(centered.T, centered) / (centered.size(0) - 1)
        mean_cov = torch.matmul(all_cluster.T, all_cluster) / (centered.size(0) - 1)
        return cov, mean_cov
    
    
    def visualize(batch_, layer_idx):
        cov, mean_cov = covariance_matrix(batch_)
        plt.figure(figsize=(10,10))
        sns.heatmap(cov, cmap="hot", annot=False)
        plt.savefig(f"interp-gains-gpt2small/plots/covariance_matrix_{layer_idx}.png")
        plt.close()
        plt.figure(figsize=(10,10))
        sns.heatmap(mean_cov, cmap='hot', annot=True, fmt='g')
        plt.savefig(f"interp-gains-gpt2small/plots/mean_covariance_matrix_{layer_idx}.png")
        plt.close()
        
        
    universal_dict = {}
    
    for files in tqdm(os.listdir('interp-gains-gpt2small/data')):
        if 'chunk0' in files:
            print(f"The filename is {files}")
            batch = torch.load(f'interp-gains-gpt2small/data/{files}')
            for k,v in batch.items():
                if k in universal_dict:
                    universal_dict[k] = torch.cat([universal_dict[k], v], dim = 0)
                else:
                    universal_dict[k] = v
            # break
    # batch = torch.load('interp-gains-gpt2small/data/chunk0_batch6.pt')
    
    for layer_idx in tqdm(range(12)):
        visualize(universal_dict[layer_idx], layer_idx)
        # _ = covariance_matrix(batch[6])




    
def main():
    
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()
    
    # get_activations(args)
    activation_analysis(args)
    
    

if __name__ == "__main__":
    main()
