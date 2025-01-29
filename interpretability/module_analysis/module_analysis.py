import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from __init__ import *



class intervention:
    def __init__(self, args):
        self.args = args
        '''
        We will be limiting our intervention on the modules for just 3 layers.
        Based on these intervention we will be building our data.
        
        It will be computed in two formats:
        
        1. Switch off 1 module M, i.e. keep 3 modules on - for one layer we can also use this, but could be minimal.
        2. Switch off 3 modules, and keep 1 module M on - if the acc is reasonable we can have this only as intervention.

        Do type 1 intervention for one layer, i.e. Layer 6.
        For many layers we can focus on type 2 intervention.
        '''
        
    
    def config_(self):
    
        with open("interpretability/module_analysis/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print(self.args.model)
        
        if self.args.model == "gpt2":
            model = HookedTransformer.from_pretrained(config[self.args.model]['model_name'])
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(config[self.args.model]['tokenizer_name'])
            model.load_state_dict(torch.load(config[self.args.model][self.args.modeltype], map_location=self.args.device, weights_only=True))
            
        else:
            model = AutoModelForCausalLM.from_pretrained(config[self.args.model][self.args.modeltype], device_map=self.args.device)
            tokenizer = AutoTokenizer.from_pretrained(config[self.args.model]['tokenizer_name'])
            # samples = pkl.load(open(config[self.args.model]['data'], "rb"))

            
        logging.basicConfig(filename = config[self.args.model]['log_file'],
                        level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info(model)
        
        
        return model, tokenizer, self.args.device, config


    
    def analysis(self, args, module, samples, config):
        
        correct = 0; total = 0
        prediction = []
        correct_samples = []
        for sample_idx in tqdm(range(len(samples))):
            sample = samples[sample_idx].to(self.device)
            predicted_string = []
            # comparing second last token of generated sentence with last token of ground truth word
            if self.args.model == "pythia70m" or "pythia1.4b":
                logits = self.model(sample)[0]
            elif self.args.model == "gpt2":
                logits = self.model(sample)
            if sample[:,-1].item() == logits[:,-2,:].argmax(dim = -1).item(): 
                prediction.append(1)
                correct_samples.append(sample)
            else:
                prediction.append(0)

        os.makedirs(f"{config[self.args.model]['data_path']}/{self.args.modeltype}", exist_ok=True)
        with open(f"{config[self.args.model]['data_path']}/{self.args.modeltype}/prediction_{self.args.type_of_intervention}_layer{self.args.num_layer}_{module}.pkl", "wb") as f:
            pkl.dump(prediction, f)
        
        with open(f"{config[self.args.model]['data_path']}/{self.args.modeltype}/samples_{self.args.type_of_intervention}_layer{self.args.num_layer}_{module}.pkl", "wb") as f:
            pkl.dump(correct_samples, f)    
    
    
    
    
    def final_analysis(self, args, config):
        
        final_dict = {}
        mean_acc = []
        
        plt.figure(figsize=(10, 5))
        
        for module in ["mod1", "mod2", "mod3", "mod4"]:
            
            with open(f"{self.config[self.args.model]['data_path']}/{self.args.modeltype}/prediction_{self.args.type_of_intervention}_layer{self.args.num_layer}_{module}.pkl", "rb") as f:
                prediction = pkl.load(f)
            
            print(prediction)
            
            final_dict[module] = prediction
            
            mean_acc.append(np.mean(prediction))
        
        plt.plot(mean_acc, marker="s", color = "orange", markersize = 10)
        plt.title("Mean Accuracy", size = 16)
        plt.xlabel("Modules", size = 12)
        plt.ylabel("Accuracy", size = 12)
        
        plt.legend()
        plt.grid(True)
        os.makedirs(f"{config[args.model]['plot_path']}/{args.modeltype}", exist_ok=True)
        plt.savefig(f"{config[args.model]['plot_path']}/{args.modeltype}/{args.type_of_intervention}_layer{args.num_layer}_accuracy.png", dpi = 300)
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
        self.list1_filtered, self.list2_filtered, self.list3_filtered, self.list4_filtered = map(list, zip(*filtered_lists))
        
        
        
    def old_graph(self):
        '''
        The visualisation of the messy graphs which has many towers and stuff.
        '''
        # Plotting
        plt.subplots(figsize=(20, 5))
        x = np.arange(len(self.list1_filtered))
        p1 = plt.bar(x, self.list1_filtered, label='Module 1', width=0.5)
        p2 = plt.bar(x, self.list2_filtered, label='Module 2', width=0.5, bottom=self.list1_filtered)
        p3 = plt.bar(x, self.list3_filtered, label='Module 3', width=0.5, bottom=np.add(self.list1_filtered, self.list2_filtered))
        p4 = plt.bar(x, self.list4_filtered, label='Module 4', width=0.5, bottom=np.add(self.list1_filtered, np.add(self.list2_filtered, self.list3_filtered)))

        plt.xlabel('Sample Index', size=12)
        plt.ylabel('Effect of Module', size=12)
        plt.title('Spike denotes when a module is turned off the accuracy for sample goes down', size=16)
        plt.legend()
        plt.grid(True)
        os.makedirs(f"{self.config[self.args.model]['plot_path']}/{self.args.modeltype}/{self.args.type_of_intervention}_layer{self.args.num_layer}_effect.png", dpi = 300)
        plt.close()
        
        
    def get_contrasting_color(self, hex_color):
        """
        Returns white for dark colors and black for light colors
        based on the brightness of the input color.
        """
        rgb = to_rgb(hex_color)  # Convert hex to RGB
        brightness = rgb_to_hsv(rgb)[2]  # Get the "value" component of HSV
        return "white" if brightness < 0.5 else "black"

    def pie_chart(self):
        """
        Creates a pie chart showing the distribution of samples depending on the number of modules they depend on.
        """
        # Initialize counters
        all_four = 0
        all_three = 0
        all_two = 0
        all_one = 0

        # Count the occurrences for each category
        for sample_idx in range(len(self.list1_filtered)):
            if self.list1_filtered[sample_idx] == self.list2_filtered[sample_idx] == self.list3_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1:
                all_four += 1
            elif (
                self.list1_filtered[sample_idx] == self.list2_filtered[sample_idx] == self.list3_filtered[sample_idx] == 1 or
                self.list1_filtered[sample_idx] == self.list2_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1 or
                self.list1_filtered[sample_idx] == self.list3_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1 or
                self.list2_filtered[sample_idx] == self.list3_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1
            ):
                all_three += 1
            elif (
                self.list1_filtered[sample_idx] == self.list2_filtered[sample_idx] == 1 or
                self.list1_filtered[sample_idx] == self.list3_filtered[sample_idx] == 1 or
                self.list1_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1 or
                self.list2_filtered[sample_idx] == self.list3_filtered[sample_idx] == 1 or
                self.list2_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1 or
                self.list3_filtered[sample_idx] == self.list4_filtered[sample_idx] == 1
            ):
                all_two += 1
            elif (
                self.list1_filtered[sample_idx] == 1 or
                self.list2_filtered[sample_idx] == 1 or
                self.list3_filtered[sample_idx] == 1 or
                self.list4_filtered[sample_idx] == 1
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
            autotext.set_color(self.get_contrasting_color(colors[i]))  # Adjust percentage text color
            autotext.set_weight("bold")  # Make percentages bold
        for text in texts:
            text.set_color("black")  # Keep labels black for consistency

        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
        plt.title("Effect of Modules on Samples", pad=20)  # Add padding to title for spacing

        # Save the figure
        output_path = f"{self.config[self.args.model]['plot_path']}/{self.args.modeltype}/pie_chart"
        os.makedirs(output_path, exist_ok=True)
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(f"{output_path}/{self.args.type_of_intervention}_layer{self.args.num_layer}_pie.png", dpi=400, bbox_inches='tight', format='png')  # High DPI for professional quality
        plt.close()
    
    
    def dataset_prepartion(self):
        correct = 0; total = 0
        samples = []
        for idx, data_ in enumerate(tqdm(make_wiki_data_loader(self.tokenizer, batch_size=self.args.batch_size), 
                                    desc="Processing batches", 
                                    total=len(make_wiki_data_loader(self.tokenizer, batch_size=self.args.batch_size)))):
            
            data = data_['tokens'].to(self.device)
            if self.args.model == "pythia70m" or "pythia1.4b":
                logits = self.model(data)[0]
            elif self.args.model == "gpt2":
                logits = self.model(data)
            if data[:,-1].item() == logits[:,-2,:].argmax(dim = -1).item():
                correct+=1
                samples.append(data)
            total+=1
            
            if idx%100 == 0:
                print(f"Accuracy: {correct/total}")
            
        self.hook_.remove()
        
        os.makedirs(f"interpretability/module_analysis/data/{self.args.model}/{self.args.modeltype}", exist_ok=True)
        with open(f"interpretability/module_analysis/data/{self.args.model}/{self.args.modeltype}/cropped_dataset_last_token.pkl", "wb") as f:
            pkl.dump(samples, f)
        
        return samples
    
    
    def hook(self, index):
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

        if self.args.model == "pythia70m" or "pythia1.4b":
            self.hook_ = self.model.gpt_neox.layers[self.args.num_layer].mlp.dense_h_to_4h.register_forward_hook(hook_fn)
        elif self.args.model == "gpt2":
            self.hook_ = self.model.transformer.h[self.args.num_layer].mlp.hook_pre.register_forward_hook(hook_fn)
    
    def forward(self, index, module, func = "analysis"):
        
        self.model, self.tokenizer, self.device, self.config = self.config_()
        self.hook(index)
        try:
            with open(f"interpretability/module_analysis/data/{self.args.model}/{self.args.modeltype}/cropped_dataset_last_token.pkl", "rb") as f:
                samples = pkl.load(f)   
        except:
            samples = self.dataset_prepartion()
            
        
        
        if func == "analysis":
            _ = self.analysis(args = self.args,
                            module = module, 
                            samples = samples, 
                            config = self.config
                            )
            
            
        elif func == "final_analysis":
            self.final_analysis(self.args, self.config)
            
            
    

def visualize(config, args):
    folder_paths = [f"{config[args.model]['plot_path']}/{args.modeltype}/pie_chart/type1",
                    f"{config[args.model]['plot_path']}/{args.modeltype}/pie_chart/type2",
                    f"{config[args.model]['plot_path']}/{args.modeltype}/pie_chart/type1",
                    f"{config[args.model]['plot_path']}/{args.modeltype}/pie_chart/type2"]
    
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


def parser_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=6)
    parser.add_argument('--type_of_intervention', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--modeltype', type=str, required=True)
    
    return parser.parse_args()



def main():
    
    args = parser_arguments()
    
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
    
    int = intervention(args)
    
    for i in tqdm(range(4)):
        if i == 0:
            print(f"Intervention using the index {i} on layer {args.num_layer}")
            int.forward(index1, module = "mod1")
        elif i == 1:
            print(f"Intervention using the index {i} on layer {args.num_layer}")
            int.forward(index2, module = "mod2")
        elif i == 2:
            print(f"Intervention using the index {i} on layer {args.num_layer}")
            int.forward(index3, module = "mod3")
        elif i == 3:
            print(f"Intervention using the index {i} on layer {args.num_layer}")
            int.forward(index4, module = "mod4")
            int.forward(index4, module="mod4", func = "final_analysis")
    
    
    # visualize(config, args)



if __name__ == '__main__':
    main()