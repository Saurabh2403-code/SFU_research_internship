import torch 
import torchvision
from tqdm import tqdm
from absl import app,flags
import os
from torchcfm.models.unet.vae import *
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from utils_cifar import ema, generate_samples, infiniteloop,logging_loss,dataloader
from tqdm import tqdm,trange
import torch.optim as optim

from PIL import Image
import torch.optim as optim

##--DEFINING FLAGS--##
FLAGS=flags.FLAGS
##model##
flags.DEFINE_integer('input_channels',3,help='Number of input channels at VAE')
flags.DEFINE_integer('hidden_channels',64,help='Number of hidden channels at next step of the input of VAE')
flags.DEFINE_integer('latent_channels',8,help='Number of channels in latent space')
flags.DEFINe_integer('batch_size',20,help='batch_size')
##trainig##
flags.DEFINE_integer('batch_size',20,help='batch_size')
flags.DEFINE_integer('epochs',5,help='epochs')
flags.Save_step('save_step',5,help='Steps after which model weight should be saved')
flags.DEFINE_float('lr',2e-4,help='learning rate')
flags.DEFINE_float('beta_target',2e-4,help='target beta value for the regularization of the latent space')
flags.DEFINE_string('savedir','/home/scratch/saurabhg/vae_models/')

device =torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def train(argv):
    def warmup(current_epoch):
        return (max(current_epoch,FLAGS.epoch)/(FLAGS.epoch))*FLAGS.beta_target
    dataset=dataloader('/home/saurabhg/scratch/LATENT_DATASET/img')
    dataloader = DataLoader(
        dataset, 
        batch_size=FLAGS.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True 
    )
    datalooper=infiniteloop(dataloader)
    model=VAE(FLAGS.input_channels,FLAGS.hidden_channels.FLAGS.latent_channels).to(device)
    model_size=0
    for param in model.parameters():
        model_size+=param.data.nelement()
    print(f"Model params: {model_size / 1000000:.2f} M")
    print(f"Model : {FLAGS.model}")
    optimizer=optim.Adam(model.parameters,lr=FLAGS.lr)
    loss_fn=nn.functional.mse_loss()
    with trange(FLAGS.epochs,dynamic_ncols=True) as pbar:
            
            
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            log_var,mean,x=model(x1)
            variance=torch.exp(log_var)
            beta=warmup(step)
            loss_recon=loss_fn(x1,x)/(2*variance)
            loss_kl=(beta*(mean@mean.T+variance+log_var-1))/2
            loss=loss_recon+loss_kl
            loss.backward()
            optimizer.step()
            if step%10==0:
                print(f'Epoch:{step} Loss:{loss.item()}')
            if step%FLAGS.Save_step==0:
                os.makedirs('FLAGS.savedir'+f'vae_model_{FLAGS.Save_step}.pt',exist_ok=True)
                torch.save(model.state_dict(),'savedir')
            
    if __name__ == "__main__":
        app.run(train)



