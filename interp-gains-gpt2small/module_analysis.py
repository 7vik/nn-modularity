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


def intervention(args, index, module, func = "analysis"):
    '''
    We will be limiting our intervention on the modules for just 3 layers.
    Based on these intervention we will be building our data.
    
    It will be computed in two formats:
    
    1. Switch off 1 module M, i.e. keep 3 modules on - for one layer we can also use this, but could be minimal.
    2. Switch off 3 modules, and keep 1 module M on - if the acc is reasonable we can have this only as intervention.

    Do type 1 intervention for one layer, i.e. Layer 6.
    For many layers we can focus on type 2 intervention.
    '''
    
    model_nmod, model_mod, tokenizer, device = config_(args)

    def hook_fn(module, input, output):
        mod_output = output.clone()
        if index == "baseline":
            return output
        else:
            if len(index) == 2:
                mod_output[:, :, index[0]:index[1]] = 0
            elif len(index) == 4:
                mod_output[:, :, index[0]:index[1]] = 0
                mod_output[:, :, index[2]:index[3]] = 0
            output = mod_output
            return output

    hook_ = model_nmod.blocks[args.num_layer].mlp.hook_pre.register_forward_hook(hook_fn)
    
    
    
    def analysis(args,module):
        
        try:
            with open(f"interp-gains-gpt2small/data/cropped_nmodel_dataset_last_token.pkl", "rb") as f:
                samples = pkl.load(f)   
        except:
            dataset_prepartion(args)
        
        correct = 0; total = 0
        prediction = []
        correct_samples = []
        for sample_idx in tqdm(range(len(samples))):
            sample = samples[sample_idx].to(device)
            logits = model_nmod(sample)
            predicted_string = []
            # comparing second last token of generated sentence with last token of ground truth word
            if sample[:,-1].item() == logits[:,-2,:].argmax(dim = -1).item(): 
                prediction.append(1)
                correct_samples.append(sample)
            else:
                prediction.append(0)

        with open(f"interp-gains-gpt2small/data/prediction_nmodel_{args.type_of_intervention}_layer{args.num_layer}_{module}.pkl", "wb") as f:
            pkl.dump(prediction, f)
        
        with open(f"interp-gains-gpt2small/data/samples_nmodel_{args.type_of_intervention}_layer{args.num_layer}_{module}.pkl", "wb") as f:
            pkl.dump(correct_samples, f)    
    
    
    def final_analysis(args):
        
        final_dict = {}
        mean_acc = []
        
        plt.figure(figsize=(10, 5))
        
        for module in ["mod1", "mod2", "mod3", "mod4"]:
            
            with open(f"interp-gains-gpt2small/data/prediction_{args.type_of_intervention}_layer{args.num_layer}_{module}.pkl", "rb") as f:
                prediction = pkl.load(f)
            
            final_dict[module] = prediction
            
            mean_acc.append(np.mean(prediction))
        
        plt.plot(mean_acc, marker="s", color = "orange", markersize = 10)
        plt.title("Mean Accuracy", size = 16)
        plt.xlabel("Modules", size = 12)
        plt.ylabel("Accuracy", size = 12)
        
        plt.legend()
        plt.grid(True)
        os.makedirs("interp-gains-gpt2small/plots/nmodel", exist_ok=True)
        plt.savefig(f"interp-gains-gpt2small/plots/nmodel/{args.type_of_intervention}_layer{args.num_layer}_accuracy.png", dpi = 300)
        plt.close()
        
        # visualize(final_dict)
        stacked_arrays = np.vstack([final_dict["mod1"],
                                    final_dict["mod2"],
                                    final_dict["mod3"],
                                    final_dict["mod4"]])
        
        
        '''
        In order to see on which samples does the model give bad accuracy
        we subtract the binary list of correct vs incorrect label with 1.
        As a result, the sentence with incorrect label has 1 and correct sentence as 0, 
        and we display the effect these incorrect 1 on graph by removing the common sentence 
        for which all the modules gave correct prediction or in other terms removing with condition:
        if (a == 0 and b == 0 and c == 0 and d == 0) then remove!
        '''
        
        filtered_lists = [
            (a, b, c, d)
            for a, b, c, d in zip(np.ones(np.array(final_dict['mod1']).shape)- final_dict['mod1'], 
                                np.ones(np.array(final_dict['mod1']).shape)- final_dict['mod2'],
                                np.ones(np.array(final_dict['mod1']).shape)- final_dict['mod3'],
                                np.ones(np.array(final_dict['mod1']).shape)- final_dict['mod4'])
            if not (a == 0 and b == 0 and c == 0 and d == 0)
        ]

        # Unzipping the filtered tuples back into separate lists
        list1_filtered, list2_filtered, list3_filtered, list4_filtered = map(list, zip(*filtered_lists))
        
        
        def old_graph():
            '''
            The visualisation of the messy graphs which has many towers and stuff.
            '''
            # Plotting
            plt.subplots(figsize=(20, 5))
            x = np.arange(len(list1_filtered))
            p1 = plt.bar(x, list1_filtered, label='Module 1', width=0.5)
            p2 = plt.bar(x, list2_filtered, label='Module 2', width=0.5, bottom=list1_filtered)
            p3 = plt.bar(x, list3_filtered, label='Module 3', width=0.5, bottom=np.add(list1_filtered, list2_filtered))
            p4 = plt.bar(x, list4_filtered, label='Module 4', width=0.5, bottom=np.add(list1_filtered, np.add(list2_filtered, list3_filtered)))

            plt.xlabel('Sample Index', size=12)
            plt.ylabel('Effect of Module', size=12)
            plt.title('Spike denotes when a module is turned off the accuracy for sample goes down', size=16)
            plt.legend()
            plt.grid(True)
            plt.savefig(f"interp-gains-gpt2small/plots/{args.type_of_intervention}_layer{args.num_layer}_effect.png", dpi = 300)
            plt.close()
        
        def get_contrasting_color(hex_color):
            """
            Returns white for dark colors and black for light colors
            based on the brightness of the input color.
            """
            rgb = to_rgb(hex_color)  # Convert hex to RGB
            brightness = rgb_to_hsv(rgb)[2]  # Get the "value" component of HSV
            return "white" if brightness < 0.5 else "black"

        def pie_chart():
            """
            Creates a pie chart showing the distribution of samples depending on the number of modules they depend on.
            """
            # Initialize counters
            all_four = 0
            all_three = 0
            all_two = 0
            all_one = 0

            # Count the occurrences for each category
            for sample_idx in range(len(list1_filtered)):
                if list1_filtered[sample_idx] == list2_filtered[sample_idx] == list3_filtered[sample_idx] == list4_filtered[sample_idx] == 1:
                    all_four += 1
                elif (
                    list1_filtered[sample_idx] == list2_filtered[sample_idx] == list3_filtered[sample_idx] == 1 or
                    list1_filtered[sample_idx] == list2_filtered[sample_idx] == list4_filtered[sample_idx] == 1 or
                    list1_filtered[sample_idx] == list3_filtered[sample_idx] == list4_filtered[sample_idx] == 1 or
                    list2_filtered[sample_idx] == list3_filtered[sample_idx] == list4_filtered[sample_idx] == 1
                ):
                    all_three += 1
                elif (
                    list1_filtered[sample_idx] == list2_filtered[sample_idx] == 1 or
                    list1_filtered[sample_idx] == list3_filtered[sample_idx] == 1 or
                    list1_filtered[sample_idx] == list4_filtered[sample_idx] == 1 or
                    list2_filtered[sample_idx] == list3_filtered[sample_idx] == 1 or
                    list2_filtered[sample_idx] == list4_filtered[sample_idx] == 1 or
                    list3_filtered[sample_idx] == list4_filtered[sample_idx] == 1
                ):
                    all_two += 1
                elif (
                    list1_filtered[sample_idx] == 1 or
                    list2_filtered[sample_idx] == 1 or
                    list3_filtered[sample_idx] == 1 or
                    list4_filtered[sample_idx] == 1
                ):
                    all_one += 1

            # Data for the pie chart
            labels = ['Depends on all 4', 'Depends on all 3', 'Depends on all 2', 'Depends on just 1']
            sizes = [all_four, all_three, all_two, all_one] # Replace with your actual data
            colors = ['#267326', '#5db85d', '#91d891', '#d4f7d4']  # Green gradient (dark to light)

            # Set up Matplotlib parameters for better styling
            mpl.rcParams.update({
                'font.size': 14,        # General font size
                'axes.titlesize': 18,   # Title size
                'axes.labelsize': 16,   # Label size
                'legend.fontsize': 12,  # Legend font size
            })

            # Create the pie chart
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',       # Show percentages with 1 decimal place
                startangle=90,          # Start from top (90 degrees)
                textprops={'fontsize': 14}  # Font size for text
            )

            # Adjust label and percentage text colors dynamically
            for i, autotext in enumerate(autotexts):
                autotext.set_color(get_contrasting_color(colors[i]))  # Adjust percentage text color
                autotext.set_weight("bold")  # Make percentages bold
            for text in texts:
                text.set_color("black")  # Keep labels black for consistency

            ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
            plt.title("Effect of Modules on Samples", pad=20)  # Add padding to title for spacing

            # Save the figure
            output_path = f"interp-gains-gpt2small/plots/pie_chart/{args.type_of_intervention}_layer{args.num_layer}_pie.png"
            plt.tight_layout()  # Adjust layout to prevent clipping
            plt.savefig(output_path, dpi=400, bbox_inches='tight', format='png')  # High DPI for professional quality
            plt.close()

        pie_chart()
        


    
    
    def dataset_prepartion(args):
        correct = 0; total = 0
        samples = []
        for idx, data_ in enumerate(tqdm(make_wiki_data_loader(tokenizer, batch_size=args.batch_size), 
                                    desc="Processing batches", 
                                    total=len(make_wiki_data_loader(tokenizer, batch_size=args.batch_size)))):
            
            data = data_['tokens'].to(device)
            logits = model_mod(data)
            if data[:,-1].item() == logits[:,-2,:].argmax(dim = -1).item():
                correct+=1
                samples.append(data)
            total+=1
            
            if idx%100 == 0:
                print(f"Accuracy: {correct/total}")
            
        hook_.remove()
        
        os.makedirs("interp-gains-gpt2small/data", exist_ok=True)
        with open(f"interp-gains-gpt2small/data/cropped_nmodel_dataset_last_token.pkl", "wb") as f:
            pkl.dump(samples, f)
        
        return correct/total
    
    
    if func == "analysis":
        _ = analysis(args,module)
        
    elif func == "final_analysis":
        final_analysis(args)
    

def visualize():
    folder_paths = ["interp-gains-gpt2small/plots/nmodel/pie_chart/type1",
                    "interp-gains-gpt2small/plots/nmodel/pie_chart/type2",
                    "interp-gains-gpt2small/plots/pie_chart/type1",
                    "interp-gains-gpt2small/plots/pie_chart/type2"]
    
    for folder_path in folder_paths:
        # List all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Optional: Sort files for consistent order
        
        # Set up the figure for a single-row layout
        num_images = len(image_files)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))  # Adjust figsize based on the number of images

        # Handle single image case (axes would not be iterable)
        if num_images == 1:
            axes = [axes]

        # Loop through images and plot each in the row
        for ax, img_file in zip(axes, image_files):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')  # Turn off axes for better appearance
            # ax.set_title(img_file, fontsize=10)  # Set the title to the file name

        # Adjust layout
        plt.tight_layout()
        
        # Save the plot to a file if specified
        plt.savefig(folder_path+"collage.png", dpi=300, bbox_inches='tight', format='png')
        

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=6)
    parser.add_argument('--type_of_intervention', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.type_of_intervention == "type1":
        # 1024//4 = 256
        index1 = [0, 256]
        index2 = [256, 256*2]
        index3 = [256*2, 256*3]
        index4 = [256*3, None]
    
    elif args.type_of_intervention == "type2":
        index1 = [256, 256*3, 256*3,  None] # switch on just 1st module
        index2 = [0, 256, 256*2, None] # switch on just 2nd module
        index3 = [0, 256*2, 256*3, None] # switch on just 3rd module
        index4 = [0, 256*3, None, None] # switch on just 4th module
        
    
    
    layer_wise_loss_dict = {}
    all_sample_loss = {}
    
    
    
    # for i in tqdm(range(4)):
    #     if i == 0:
    #         print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index1, module = "mod1")
    #     elif i == 1:
    #         print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index2, module = "mod2")
    #     elif i == 2:
    #         print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index3, module = "mod3")
    #     elif i == 3:
    #         print(f"Intervention using the index {i} on layer {args.num_layer}")
    #         intervention(args, index4, module="mod4")
    #         intervention(args, index4, module="mod4", func = "final_analysis")
    visualize()



if __name__ == '__main__':
    main()