"""
this file corntains the model class for variational Auto encoder"""
import torch
import torch.nn as nn

class VAE_Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            nn.MaxPool2d(2), # 128 -> 64
            
            nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1),
            ResidualBlock(hidden_channels*2),
            nn.MaxPool2d(2), # 64 -> 32
            
            nn.Conv2d(hidden_channels*2, output_channels, 3, padding=1),
            ResidualBlock(output_channels),
            nn.MaxPool2d(2)  # 32 -> 16
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x + self.block(x)) # Identity skip connection

class VAE_Latent(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    channel=int(x.shape[1]/2)
    mean=x[:,:channel,:,:]
    log_var=x[:,channel:,:,:]
    std=torch.exp(0.5*log_var)
    noise=torch.randn_like(std)
    sampled_z=mean+torch.mul(std,noise)
    return log_var,mean,sampled_z


class VAE_Decoder(nn.Module):
  def __init__(self, in_channels, hidden_channels):
    super().__init__()
    # Starting from 16x16 latent
    self.decoder_layer = nn.Sequential(
        # Level 1: 16x16 -> 32x32
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
        ResidualBlock(hidden_channels), # Added to preserve latent detail
        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
        nn.SiLU(),

        # Level 2: 32x32 -> 64x64
        ResidualBlock(hidden_channels),
        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
        nn.SiLU(),

        # Level 3: 64x64 -> 128x128
        ResidualBlock(hidden_channels // 2),
        nn.ConvTranspose2d(hidden_channels // 2, 3, kernel_size=4, stride=2, padding=1),
        nn.Tanh() # Projects back to [-1, 1] range for images
    )

  def forward(self, x):
    return self.decoder_layer(x)

class VAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder=VAE_Encoder(3,64,8)
    self.latent_sampler=VAE_Latent()
    self.decoder=VAE_Decoder(4,256)
  def forward(self,x):
    # print(f'input {x.shape}')
    x=self.encoder(x)
    # print(f'encoded {x.shape}')
    log_var,mean,latent_image=self.latent_sampler(x)
    # print(f'latent_vector {latent_vector.shape} latent image {latent_image.shape}')
    x=self.decoder(latent_image)
    # print(f'decoded {x.shape}')
    return log_var,mean,x