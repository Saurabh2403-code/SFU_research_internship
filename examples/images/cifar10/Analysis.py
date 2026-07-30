import os
import sys
import copy
import torch
from torchvision import datasets,transforms
from absl import app,flags
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
parent_folder= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder)
from utils_cifar import *
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'
FLAGS=flags.FLAGS
flags.DEFINE_integer('num_images',100,help='Number Of images to generate')
flags.DEFINE_integer('num_channels',128,help='Number Of Base Colour Channels')
flags.DEFINE_string('output_dir','/scratch/saurabhg/flow_outputs/',help='Ouput Directory Address')
flags.DEFINE_string('model','otcfm',help='model_type')
flags.DEFINE_integer('step',15000,help='Epoch number after which we are evaluating the model')
flags.DEFINE_integer('time_steps',1,help='time_steps_to_simulate_ode')
flags.DEFINE_bool('parallel',False,help='Multi GPU training')
def inference(argv):

    net_model=UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)


    model_dir=FLAGS.output_dir+f'{FLAGS.model}'+'/'
    save_dir=model_dir+'distances'+'/'
    os.makedirs(save_dir,exist_ok=True)
    heatmap_dir=save_dir+f'{FLAGS.model}_{FLAGS.num_images}'
    os.makedirs(heatmap_dir,exist_ok=True)
    state_dict=torch.load(model_dir+f'{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt', weights_only=True)
    net_model.load_state_dict(state_dict["net_model"])
    net_model.eval()
    with torch.no_grad():
        generated_samples=generate_samples(net_model, FLAGS.parallel,model_dir+f"{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt",step=FLAGS.step,time_steps=FLAGS.time_steps,number_of_images=FLAGS.num_images,net_="normal")
        original_dataset=get_original_image(FLAGS.num_images)
        l2_distance=get_l2_distance(generated_samples,original_dataset)
        torch.save(l2_distance,save_dir+f'{FLAGS.model}_Distances_{FLAGS.num_images}.pt')
        plt.imshow(l2_distance.detach().cpu().numpy(),origin='lower')
        plt.colorbar()
        plt.savefig(os.path.join(heatmap_dir,'Distances_Heat_Map.png'),bbox_inches='tight')
        plt.close()
if __name__=="__main__":
    app.run(inference)





