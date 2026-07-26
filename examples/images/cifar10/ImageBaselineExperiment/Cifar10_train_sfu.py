import os
import sys
import copy
import torch
from torchvision import datasets,transforms
from absl import app,flags
from tqdm import trange
parent_folder=parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder)
from utils_cifar import ema, generate_samples, infiniteloop
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS=flags.FLAGS

flags.DEFINE_string('model','otcfm',help='model type')
flags.DEFINE_string('output_dir','/scratch/saurabhg/flow_outputs/',help='Ouput directory Address')
#unet

flags.DEFINE_integer('num_channels',32,help='Base color channels in UNET')



#training
flags.DEFINE_float('lr',2e-4,help='Learining Rate')
flags.DEFINE_integer('epochs',20000,help='Number of epochs for training')
flags.DEFINE_float('grad_clip',1.0,help='gradient clipping norm')
flags.DEFINE_integer('lr_warmup',50,help='learning rate warmup')
flags.DEFINE_integer('batch_size',128,help='batch_size')
flags.DEFINE_integer('num_workers',4,help='Number Of Dataloader Worker')
flags.DEFINE_float('ema_decay',0.99,help='ema_decay_rate')
flags.DEFINE_bool('parallel',False,help='Multi gpu training')
#EVALUATION
flags.DEFINE_integer('Save_step',5000,help='Epochs after which model is saved')
flags.DEFINE_bool('Dataset_download_flag',False,help='Do you want to download data or not?')
flags.DEFINE_integer('tiny_dataset_size',100,help='Number of images for the Dataset Subset')
flags.DEFINE_string('dataset_adress','/scratch/saurabhg/cifar10_data/',help='Adress where dataset will be stored')
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
    dataset=datasets.CIFAR10(
        root=FLAGS.dataset_adress,
        train=True,
        download=FLAGS.Dataset_download_flag,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )
    )
    indices=list(range(FLAGS.tiny_dataset_size))
    tiny_dataset=torch.utils.data.Subset(dataset,indices)
  
    dataloader=torch.utils.data.DataLoader(
        tiny_dataset,
        num_workers=FLAGS.num_workers,
        batch_size=FLAGS.batch_size,
        shuffle=True,
    )
    datalooper=infiniteloop(dataloader)


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


    ema_model=copy.deepcopy(net_model)
    optim=torch.optim.Adam(net_model.parameters(),lr=FLAGS.lr)
    sched=torch.optim.lr_scheduler.LambdaLR(optim,lr_lambda=warmup_lr)

    if FLAGS.parallel:
        print('Warning: Using multi gpu training can deteriorate the performance of the model if possible try to train on a single GPU, training on single GPU requires upto 8gb of memory')
        net_model=torch.nn.DataParallel(net_model)
        ema_model=torch.nn.DataParallel(ema_model)
    # show Model_size

    model_size=0
    for param in net_model.parameters():
        model_size+=param.data.nelement()
    print(f"Model params: {model_size / 1000000:.2f} M")

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

            # sample and Saving the weights
            if FLAGS.Save_step > 0 and step % FLAGS.Save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
