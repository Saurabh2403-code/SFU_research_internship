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

flags.DEFINE_integer('num_images', 200, help='Total number of images to generate')
flags.DEFINE_integer('batch_size', 100, help='Batch size to prevent CUDA OOM')
flags.DEFINE_integer('num_channels', 128, help='Number Of Base Colour Channels')
flags.DEFINE_string('output_dir', '/scratch/saurabhg/flow_outputs/', help='Output Directory Address')
flags.DEFINE_string('model', 'icfm', help='model_type')
flags.DEFINE_integer('step', 90000, help='Epoch number after which we are evaluating the model')
flags.DEFINE_integer('time_steps', 100, help='time_steps_to_simulate_ode')
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
        state_dict = torch.load(model_dir + f'{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt', map_location=device, weights_only=True)
        net_model.load_state_dict(state_dict["net_model"])
        net_model.eval()

        print(f"Generating {FLAGS.num_images} images on {device}...")
        samples = generate_samples(
            net_model, 
            FLAGS.parallel,
            model_dir + f"{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt",
            step=FLAGS.step,
            time_steps=FLAGS.time_steps,
            number_of_images=FLAGS.num_images,
            net_="normal"
        )
        original_dataset = get_original_image(100).cpu()

if __name__ == "__main__":
    app.run(inference)