import os
import sys
import torch
from absl import app, flags
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))

cifar10_dir = os.path.join(os.path.dirname(current_dir), 'cifar10')

sys.path.append(cifar10_dir)

from utils_cifar import *

from torchcfm.models.unet.unet import UNetModelWrapper
from diffusers.models import AutoencoderKL
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_images', 1000, help='Total number of images to generate')
flags.DEFINE_integer('batch_size', 50, help='Batch size to prevent CUDA OOM')
flags.DEFINE_integer('num_color_channels',4,help='number of color channels for the input image to the unet')
flags.DEFINE_integer('num_channels', 128, help='Number Of Base Colour Channels')
flags.DEFINE_string('output_dir', '/scratch/saurabhg/LATENT_DATASET/flow_outputs/', help='Output Directory Address')
flags.DEFINE_string('model', 'icfm', help='model_type')
flags.DEFINE_integer('step', 80000, help='Epoch number after which we are evaluating the model')
flags.DEFINE_integer('time_steps', 100, help='time_steps_to_simulate_ode')
flags.DEFINE_bool('parallel', False, help='Multi GPU training')

def inference(argv):
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae=vae.to(device)

        net_model = UNetModelWrapper(
            dim=(4, 32, 32),
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
        state_dict = torch.load(model_dir + f'{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt', map_location=device, weights_only=True)
        net_model.load_state_dict(state_dict["net_model"])
        net_model.eval()
        all_generated=[]
        print(f"Generating {FLAGS.num_images} images in batches of {FLAGS.batch_size} on {device}...")
        for i in range(FLAGS.num_images//FLAGS.batch_size):

            samples = generate_samples(
                net_model, 
                FLAGS.parallel,
                model_dir + f"{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt",
                step=FLAGS.step,
                time_steps=FLAGS.time_steps,
                number_of_images=FLAGS.batch_size,
                net_="normal"
            )
            scaled_latent=samples/(vae.config.scaling_factor)
            decoded_image=vae.decode(scaled_latent)
            decoded_image=decoded_image.sample
            all_generated.append(decoded_image.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        generated_samples=torch.cat(all_generated,dim=0)
        generated_samples=generated_samples*0.5+0.5
        generated_samples=generated_samples.clip(0,1)
        torch.save(generated_samples,f'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/image_tensor/generated_images_{FLAGS.num_images}_iteration3_python_inference_scipt.pt')
        torchvision.utils.save_image(generated_samples,f'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/generated_images/generated_images_{FLAGS.num_images}_{FLAGS.step}_python_inference_script_iteration3.png')
        

if __name__ == "__main__":
    app.run(inference)