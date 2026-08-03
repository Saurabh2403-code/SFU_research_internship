import copy
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE
import torchvision
# from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from absl import flags
from pathlib import Path
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
FLAGS=flags.FLAGS
flags.DEFINE_bool('return_image_tensor',True,help='Whether do you want to get the tensor of image or not')
def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def generate_samples(model, parallel, savedir, step,time_steps:int=1,number_of_images:int=64,net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(number_of_images, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, time_steps+1, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=10)

    model.train()
    if FLAGS.return_image_tensor:
        return traj


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

            
def logging_loss(loss_val, model_name, loss_file='/scratch/saurabhg/losses/'):
    os.makedirs(loss_file, exist_ok=True)
    with open(os.path.join(loss_file, f'{model_name}.txt'), "a") as file:
        file.write(f"{loss_val}\n")





def plot_loss(
    loss_file: str = "/scratch/saurabhg/losses", model: str = "otcfm"
):

    base_dir = Path(loss_file)
    save_dir = base_dir / f"{model}_loss_plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    source_file = base_dir / f"{model}.txt"
    data = np.loadtxt(source_file)

    plt.figure()  
    plt.plot(data)
    plt.title(f"{model.upper()} Training Loss")
    plt.xlabel("Epochs / Iterations")
    plt.ylabel("Loss")

    plt.savefig(save_dir / "loss_plot.png", dpi=150, bbox_inches="tight")
    plt.close() 



def get_original_image(count:int=100):
     dataset=torchvision.datasets.CIFAR10(
          root='/scratch/saurabhg/cifar10_data',
          download=False,
          transform=torchvision.transforms.Compose(
               [
                torchvision.transforms.ToTensor(),  
                torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))    
               ]
          )
     )
     indices=list(range(count))
     tiny_dataset=Subset(dataset,indices)
     dataloader=DataLoader(
          tiny_dataset,
          shuffle=False,
          batch_size=count     
     )
     data=next(iter(dataloader))[0]
     data=data*0.5+0.5
     savedir="scratch/saurabhg/cifar10_data"
     os.makedirs(savedir,exist_ok=True)


     save_image(data,f'{savedir}/original_image_{count}.png')
     if FLAGS.return_image_tensor:
         return data

def get_l2_distance(original_dataset,generated_dataset):
    """
    Input:original_dataset.shape=[B,H,W,3],generated_dataset.shape=[B,H,W,3]
    """
    original_dataset=torch.flatten(original_dataset,start_dim=1) #[B,H*W*3]
    generated_dataset=torch.flatten(generated_dataset,start_dim=1)
    return torch.cdist(original_dataset.to(device),generated_dataset.to(device),p=2)
     
def detect_mode_collapse(distance_matrix):
    distance_matrix=distance_matrix if isinstance(distance_matrix,torch.tensor) else torch.tensor(distance_matrix)
    closest_match=torch.argmin(distance_matrix,dim=1)
    return len(torch.unique(closest_match))/distance_matrix.shape[0]


    
