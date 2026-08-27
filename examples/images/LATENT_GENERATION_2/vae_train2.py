import torch
import torchvision
import os
import sys
import copy
current_dir = os.path.dirname(os.path.abspath(__file__))

cifar10_dir = os.path.join(os.path.dirname(current_dir), 'cifar10')

sys.path.append(cifar10_dir)

from utils_cifar import *


from diffusers.models import AutoencoderKL
from absl import flags,app
from tqdm import tqdm,trange
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder)
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper



print('Module Imported')
FLAGS=flags.FLAGS

##DATASET##
flags.DEFINE_string('dataset_address','/home/saurabhg/scratch/CORGIS_DATASET/img',help='dataset_address')

flags.DEFINE_string('model','icfm',help='model type')
flags.DEFINE_string('output_dir','/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs/',help='Ouput directory Address')

flags.DEFINE_integer('num_channels',128,help='Base color channels in UNET')
flags.DEFINE_integer('num_color_channels',4,help='number of color channels for the input image to the unet')
flags.DEFINE_float('lr',2e-4,help='Learining Rate')
flags.DEFINE_integer('epochs',100000,help='Number of epochs for training')
flags.DEFINE_float('grad_clip',1.0,help='gradient clipping norm')
flags.DEFINE_integer('lr_warmup',1,help='learning rate warmup')
flags.DEFINE_integer('batch_size',100,help='batch_size')
flags.DEFINE_integer('num_workers',4,help='Number Of Dataloader Worker')
flags.DEFINE_float('ema_decay',0.99,help='ema_decay_rate')
flags.DEFINE_bool('parallel',False,help='Multi gpu training')
#EVALUATION
flags.DEFINE_integer('Save_step',5000,help='Epochs after which model is saved')
flags.DEFINE_bool('Dataset_download_flag',False,help='Do you want to download data or not?')

flags.DEFINE_integer('time_steps',100,help='time_steps_to_simulate_ode')


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def warmup_lr(step):
    return min(step,FLAGS.lr_warmup)/FLAGS.lr_warmup

def train(argv):
    print(
            "lr,batch_size,training_epochs,num_workers:",
              FLAGS.lr,
              FLAGS.batch_size,
              FLAGS.epochs,
              FLAGS.num_workers
        )


    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae=vae.to(device)
    vae.eval()
    dataset=FlatImageDataset(FLAGS.dataset_address)
    full_dataset_size=len(dataset)
    print("Pre-encoding dataset into latents...")
    encode_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    latents_list = []
    
    with torch.no_grad():
        for batch in tqdm(encode_loader, desc="Encoding Images"):
            batch = batch.to(device)
            dist = vae.encode(batch).latent_dist
            latents_list.append(dist.sample()) 
            
    encoded_dataset = torch.cat(latents_list, dim=0)
    print(f"Finished encoding. Latent dataset shape: {encoded_dataset.shape}")
    latents = (encoded_dataset * vae.config.scaling_factor).detach()


    print("Generating the massive 1,300-image grid...")
    full_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    all_images = next(iter(full_loader))
  
    os.makedirs('/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs/icfm/image_tensor/', exist_ok=True)
    torch.save(all_images * 0.5 + 0.5, '/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs/icfm/image_tensor/original_image_tensor.pt')
    torchvision.utils.save_image(all_images * 0.5 + 0.5, '/home/saurabhg/scratch/CORGIS_DATASET/original_image.png', nrow=50)

    latent_dataset = torch.utils.data.TensorDataset(latents)
    latent_dataloader = torch.utils.data.DataLoader(latent_dataset, batch_size=FLAGS.batch_size, shuffle=True)


    datalooper=infiniteloop(latent_dataloader)

    net_model=UNetModelWrapper(
            dim=(FLAGS.num_color_channels, 32, 32),
            num_res_blocks=2,
            num_channels=FLAGS.num_channels,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(device)


    ema_model=copy.deepcopy(net_model)
    optim=torch.optim.Adam(net_model.parameters(),lr=FLAGS.lr)
    sched=torch.optim.lr_scheduler.LambdaLR(optim,lr_lambda=warmup_lr)
    
    if FLAGS.parallel:
        print('Warning: Using multi gpu training can deteriorate the performance of the model if possible try to train on a single GPU, training on single GPU requires upto 8gb of memory')
        net_model=torch.nn.DataParallel(net_model)
        ema_model=torch.nn.DataParallel(ema_model)
    model_size=0
    for param in net_model.parameters():
        model_size+=param.data.nelement()
    print(f"Model params: {model_size / 1000000:.2f} M")
    print(f"Model : {FLAGS.model}")

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )
    savedir=FLAGS.output_dir+FLAGS.model+"/"

    os.makedirs(savedir,exist_ok=True)

    with trange(FLAGS.epochs,dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()

            x1 = next(datalooper).to(device)

            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)

            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            if step%100==0:
                logging_loss(loss.item(),FLAGS.model)
                
            # sample and Saving the weights
            if FLAGS.Save_step > 0 and (step+1) % FLAGS.Save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step,time_steps=FLAGS.time_steps,net_="normal",save_generated_images=True)
                generate_samples(ema_model, FLAGS.parallel, savedir, step,time_steps=FLAGS.time_steps,net_="ema",save_generated_images=True)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step+1}.pt",
                )
    
    
if __name__ == "__main__":
    app.run(train)
    
    



