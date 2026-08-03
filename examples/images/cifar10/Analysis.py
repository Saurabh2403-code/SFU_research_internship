import os
import sys
import torch
from absl import app, flags
from tqdm import tqdm

parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder)
from utils_cifar import *
from torchcfm.models.unet.unet import UNetModelWrapper

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_images', 1000, help='Total number of images to generate')
flags.DEFINE_integer('batch_size', 100, help='Batch size to prevent CUDA OOM')
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
        
        # FIX: map_location securely loads the weights whether on CPU or GPU
        state_dict = torch.load(model_dir + f'{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt', map_location=device, weights_only=True)
        net_model.load_state_dict(state_dict["net_model"])
        net_model.eval()

        print(f"Generating {FLAGS.num_images} images in batches of {FLAGS.batch_size} on {device}...")
        all_generated = []
        
        for _ in range(FLAGS.num_images // FLAGS.batch_size):
            samples = generate_samples(
                net_model, 
                FLAGS.parallel,
                model_dir + f"{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt",
                step=FLAGS.step,
                time_steps=FLAGS.time_steps,
                number_of_images=FLAGS.batch_size,
                net_="normal"
            )
            all_generated.append(samples.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
            
        generated_samples = torch.cat(all_generated, dim=0)
        original_dataset = get_original_image(100).cpu()

        print("Calculating L2 distances...")
        l2_matrix = torch.zeros(100, FLAGS.num_images)

        for i in tqdm(range(100), desc="Comparing against original images"):
            orig_img = original_dataset[i].unsqueeze(0).to(device)
            
            for j in range(0, FLAGS.num_images, FLAGS.batch_size):
                gen_batch = generated_samples[j:j+FLAGS.batch_size].to(device)
                
                l2_dist = torch.mean((orig_img - gen_batch) ** 2, dim=[1, 2, 3])
                l2_matrix[i, j:j+FLAGS.batch_size] = l2_dist.cpu()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_path = save_dir + f'{FLAGS.model}_Distances_{FLAGS.num_images}.pt'
        torch.save(l2_matrix, save_path)
        print(f"Distance matrix successfully saved to {save_path}!")

if __name__ == "__main__":
    app.run(inference)