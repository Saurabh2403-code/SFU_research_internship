import os
import sys
import copy
import torch
from torchvision import datasets, transforms
from absl import app, flags
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder)
from utils_cifar import *
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_images', 10000, help='Total number of images to generate')
flags.DEFINE_integer('batch_size', 100, help='Batch size to prevent CUDA OOM') # <-- Added Batch Size
flags.DEFINE_integer('num_channels', 128, help='Number Of Base Colour Channels')
flags.DEFINE_string('output_dir', '/scratch/saurabhg/flow_outputs/', help='Output Directory Address')
flags.DEFINE_string('model', 'otcfm', help='model_type')
flags.DEFINE_integer('step', 15000, help='Epoch number after which we are evaluating the model')
flags.DEFINE_integer('time_steps', 1, help='time_steps_to_simulate_ode')
flags.DEFINE_bool('parallel', False, help='Multi GPU training')

def inference(argv):
    with torch.no_grad():
        net_model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=FLAGS.num_channels,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(device)

        model_dir = FLAGS.output_dir + f'{FLAGS.model}/'
        save_dir = model_dir + 'distances/'
        os.makedirs(save_dir, exist_ok=True)
        heatmap_dir = save_dir + f'{FLAGS.model}_{FLAGS.num_images}'
        os.makedirs(heatmap_dir, exist_ok=True)
        
        state_dict = torch.load(model_dir + f'{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt', weights_only=True)
        net_model.load_state_dict(state_dict["net_model"])
        net_model.eval()

        print(f"Generating {FLAGS.num_images} images in batches of {FLAGS.batch_size}...")
        all_generated = []
        
        for _ in range(FLAGS.num_images // FLAGS.batch_size):
            samples = generate_samples(
                net_model, 
                FLAGS.parallel,
                model_dir + f"{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt",
                step=FLAGS.step,
                time_steps=FLAGS.time_steps,
                number_of_images=FLAGS.batch_size, # <-- Pass batch_size to generation
                net_="normal"
            )
            # Move to CPU immediately to free VRAM
            all_generated.append(samples.cpu())
            torch.cuda.empty_cache() 
            
        generated_samples = torch.cat(all_generated, dim=0)
        original_dataset = get_original_image(100).cpu()

        print("Calculating L2 distances...")
        
        l2_matrix = torch.zeros(100, FLAGS.num_images)


        for i in tqdm(range(100), desc="Comparing against original images"):
            orig_img = original_dataset[i].unsqueeze(0).to(device)
            
            for j in range(0, FLAGS.num_images, FLAGS.batch_size):
                gen_batch = generated_samples[j:j+FLAGS.batch_size].to(device)
                
                # Calculate L2 Distance
                l2_dist = torch.mean((orig_img - gen_batch) ** 2, dim=[1, 2, 3])
                l2_matrix[i, j:j+FLAGS.batch_size] = l2_dist.cpu()

                torch.cuda.empty_cache()

        torch.save(l2_matrix, save_dir + f'{FLAGS.model}_Distances_{FLAGS.num_images}.pt')

        # Plot L2 Heatmap
        plt.imshow(l2_matrix.numpy(), origin='lower', aspect='auto')
        plt.colorbar()
        plt.title("L2 Pixel Distance")
        plt.savefig(os.path.join(heatmap_dir, 'Distances_Heat_Map.png'), bbox_inches='tight')
        plt.close()

        print("Inference and evaluation complete!")

if __name__ == "__main__":
    app.run(inference)