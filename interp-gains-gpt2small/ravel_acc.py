from __init__ import *

device = torch.device("mps")  

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")
model.to(device)
# model.load_state_dict(torch.load('wiki_fully_modular_mlp_in_out.pt', map_location=device))

nmodel = LanguageModel("openai-community/gpt2", device_map=device)

with open("interp-gains-gpt2small/data/comfy_continent.json", "rb") as f:
    data = json.load(f)
    
# Finding the accuracy difference of the openai-community model and the gpt2-small model

def huggingface_gpt2_acc():

    total = 0
    correct = 0
    for sample in tqdm(data):
        input, target = sample[0], sample[1]
        
        with nmodel.trace(input) as tracer:
            intervened_base_predicted = nmodel.lm_head.output.argmax(dim=-1).save()
            intervened_base_output = nmodel.lm_head.output.save()

        predicted_text = []
        for index in range(intervened_base_output.shape[0]):
            predicted_text.append(
                nmodel.tokenizer.decode(
                    intervened_base_output[index].argmax(dim=-1)
                ).split()[-1]
            )
        if predicted_text[0] == target.split()[0]:
            correct+=1
            total+=1
        else: total+=1
    
    return correct/total


def gpt2small_acc():
    
    total_tlens = 0
    correct_tlens = 0
    for sample in tqdm(data):
        input, target = sample[0], sample[1]

        tokens = model.to_tokens(input)
        logits, cache = model.run_with_cache(tokens)
        a = logits[:,-1,:].argmax(dim=-1)
        predicted = model.to_string(a)
        if predicted.split()[0] == target.split()[0]:
            correct_tlens+=1
            total_tlens+=1
        else: total_tlens+=1
    
    return correct_tlens/total_tlens
    

def ful_modular_acc():
        
    total_tlens = 0
    correct_tlens = 0
    model.load_state_dict(torch.load('interp-gains-gpt2small/data/wiki_fully_modular_mlp_in_out.pt', map_location=device))
    for sample in tqdm(data):
        input, target = sample[0], sample[1]
        tokens = model.to_tokens(input)
        logits, cache = model.run_with_cache(tokens)
        a = logits[:,-1,:].argmax(dim=-1)
        predicted = model.to_string(a)
        if predicted.split()[0] == target.split()[0]:
            correct_tlens+=1
            total_tlens+=1
        else: total_tlens+=1
    
    return correct_tlens/total_tlens


    
def non_modular_acc():
        
    total_tlens = 0
    correct_tlens = 0
    model.load_state_dict(torch.load('interp-gains-gpt2small/data/wiki_non_modular_mlp_in_out.pt', map_location=device))
    for sample in tqdm(data):
        input, target = sample[0], sample[1]
        tokens = model.to_tokens(input)
        logits, cache = model.run_with_cache(tokens)
        a = logits[:,-1,:].argmax(dim=-1)
        predicted = model.to_string(a)
        if predicted.split()[0] == target.split()[0]:
            correct_tlens+=1
            total_tlens+=1
        else: total_tlens+=1
    
    return correct_tlens/total_tlens
    

if __name__ == "__main__":
    acc1 = huggingface_gpt2_acc()
    acc2 = gpt2small_acc()
    acc3 = ful_modular_acc()
    acc4 = non_modular_acc()
    
    print()
    print("Hugging Face Acc \t GPT2-Small Acc \t Fully Modular Acc \t Non Modular Acc")
    print(f"{acc1} \t  \t  {acc2} \t \t {acc3} \t \t {acc4}")