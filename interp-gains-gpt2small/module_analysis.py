from __init__ import *


def config_(args):
    model =  GPT2Model.from_pretrained('gpt2')
    
    model_nmod = HookedTransformer.from_pretrained('gpt2-small')
    model_mod = HookedTransformer.from_pretrained('gpt2-small')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device(args.device)

    model_nmod.load_state_dict(torch.load('interp-gains-gpt2small/data/models/wiki_non_modular_mlp_in_out.pt', map_location=device, weights_only=True))
    model_mod.load_state_dict(torch.load('interp-gains-gpt2small/data/models/wiki_fully_modular_mlp_in_out.pt', map_location=device, weights_only=True))

    logging.basicConfig(filename = 'interp-gains-gpt2small/logs/model_analysis.log',
                        level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    model_nmod.to(device)
    model_mod.to(device)
    logging.info(model_nmod)
    
    return model_nmod, model_mod, tokenizer, device


def nps(arr_):
    pprint(np.array(arr_).shape)


def intervention(args, index):
    '''
    We will be limiting our intervention on the modules for just 3 layers.
    Based on these intervention we will be building our data.
    
    It will be computed in two formats:
    
    1. Switch off 1 module M, i.e. keep 3 modules on - for one layer we can also use this, but could be minimal.
    2. Switch off 3 modules, and keep 1 module M on - if the acc is reasonable we can have this only as internvention.

    Do type 1 intervention for one layer, i.e. Layer 6.
    For many layers we can focus on type 2 intervention.
    
    #TODO: Make graphs per data point also. if it does not happen then we can have a single graph showcasign the overlapping data for which the output was wrong.
    #TODO: Make bar graph of accuracy for each type of intervention on each module.
    #TODO: Repeat the above process for type 2 intervention.
    '''
    
    model_nmod, model_mod, tokenizer, device = config_(args)
    
    def hook_fn(module, input, output):
        mod_output = output.clone()
        if index == "baseline":
            return output
        else:
            mod_output[:, :, index[0]:index[1]] = 0
            output = mod_output
            return output

    hook_ = model_mod.blocks[args.num_layer].mlp.hook_pre.register_forward_hook(hook_fn)
    
    
    
    def analysis():
        with open("interp-gains-gpt2small/data/cropped_dataset_topk1000_acc.pkl", "rb") as f:
            samples = pkl.load(f)   
        
        correct = 0; total = 0
        for sample_idx in tqdm(range(len(samples))):
            sample = samples[sample_idx].to(device)
            logits = model_mod(sample)
            # if int(sample[:,-1].item()) in [int(x) for x in logits[:,-1,:].topk(1000, dim = -1).indices.tolist()[0]]:
            #     correct += 1
            # total+=1

            # print(tokenizer.decode(sample[0].tolist()))
            # print(f"\n\n\n\n")
            predicted_string = []
            # Decode each sequence individually
            # for seq in logits.argmax(dim=-1).tolist():
            #     predicted_string.append(tokenizer.decode(seq))
            # print(" ".join(predicted_string))
            # equals = (sample[:,1:].to("cpu") == logits[:,1:-1,:].argmax(dim = -1).to("cpu")).float().mean()  # -> method of Satvik.
            if sample[:,-1].item() == logits[:,-2,:].argmax(dim = -1).item(): # -> my improved method.
                correct += 1
            total+=1
            if sample_idx%100 == 0 and sample_idx != 0:
                print(f"Accuracy: {correct/total}")    
        return correct/total
    
    
    
    
    def dataset_prepartion(args):
        correct = 0; total = 0
        samples = []
        for idx, data_ in enumerate(tqdm(make_wiki_data_loader(tokenizer, batch_size=args.batch_size), 
                                    desc="Processing batches", 
                                    total=len(make_wiki_data_loader(tokenizer, batch_size=args.batch_size)))):
            
            data = data_['tokens'].to(device)
            logits = model_mod(data)
            # print([int(x) for x in logits[:,-1,:].topk(3, dim = -1).indices.tolist()[0]])
            # print(int(data[:,-1].item()))
            # if data[:,-1].item() == logits.argmax(dim = -1).item():
            if data[:,-1].item() == logits[:,-2,:].argmax(dim = -1).item():
                correct+=1
                samples.append(data)
            total+=1
            

            '''
            The loss should be calculated as binary and only on last token.
            The accuracy of the model_mod is 1.5% on predicting the last token.
            So i figured to get the accuracy on top 3,5, and 10 tokens. 
            The accuracy for them are as follows:
            Top 1: 1.5% argmax() 
            Top 3: 2.4% 
            Top 100: 12%
            Top 1000: 54%
            
            '''
            # all_loss.append(loss.item())
            if idx%100 == 0:
                print(f"Accuracy: {correct/total}")
            
        hook_.remove()
        
        os.makedirs("interp-gains-gpt2small/data", exist_ok=True)
        with open(f"interp-gains-gpt2small/data/cropped_dataset_last_token_layer{args.num_layer}.pkl", "wb") as f:
            pkl.dump(samples, f)
        
        return correct/total

    # analysis()
    accuracy = dataset_prepartion(args)
    print(f"The accuracy of the trained model is {accuracy}")
    

def visualize(dictionary):
    plt.figure(figsize=(10, 5))
    plt.plot(dictionary[0], label='Layer 1')
    plt.plot(dictionary[1], label='Layer 2')
    plt.plot(dictionary[2], label='Layer 3')
    plt.plot(dictionary[3], label='Layer 4')
    plt.plot(dictionary["baseline"], label='Baseline', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.show()
    plt.close()


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=6)
    
    args = parser.parse_args()
    
    index1 = [0, 1024//4]
    index2 = [1024//4, 1024//2]
    index3 = [1024//2, (1024//4)*3]
    index4 = [(1024//4)*3, 1024]
    
    layer_wise_loss_dict = {}
    all_sample_loss = {}
    
    all_sample_loss["baseline"] = intervention(args, "baseline")
    '''
    We should only process the dataset for which the trained model produces accurate output.
    '''
    
    # for i in tqdm(range(4)):
    #     if i == 0:
            # print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index1)
    #     elif i == 1:
            # print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index2)
    #     elif i == 2:
            # print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index3)
    #     elif i == 3:
            # print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index4)
    
    # Focusing Layers: 2,5,6,7,10
    # for layer_idx in tqdm(range(12)):
    #     args.num_layer = layer_idx
    #     all_loss = intervention(args, index1)
        # layer_wise_loss_dict[layer_idx] = np.mean(np.array(all_loss))
    
    # pprint(all_sample_loss)
    
    # visualize(all_sample_loss)
    


if __name__ == '__main__':
    main()