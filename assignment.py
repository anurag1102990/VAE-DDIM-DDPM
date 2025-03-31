import torch

from models import VAE
from models import DDPM
from models import DDIM
from models import LDDPM
from models import UNet
from models import VarianceScheduler


def prepare_ddpm() -> DDPM:
    """
    EXAMPLE OF INITIALIZING DDPM. Feel free to change the following based on your needs and implementation.
    """
    # Define the configurations of the Variance Scheduler
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # Dfine the confifurations of the UNet
    in_channels = 1
    down_channels=[64, 128, 128, 128]
    up_channels = [128, 128, 128, 64]
    time_embed_dim = 128
    num_classes = 10

    # TODO: define the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # TODO: define the noise estimating UNet
    network = UNet(in_channels=in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)
    
    ddpm = DDPM(network=network, var_scheduler=var_scheduler)

    return ddpm

def prepare_ddim() -> DDIM:
    """
    EXAMPLE OF INITIALIZING DDIM. Feel free to change the following based on your needs and implementation.
    """
    # Define the configurations of the Variance Scheduler
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # Define the confifurations of the UNet
    in_channels = 1
    down_channels=[64, 128, 128, 128]
    up_channels = [128, 128, 128, 64]
    time_embed_dim = 128
    num_classes = 10

    # Define the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # Define the noise estimating UNet
    network = UNet(in_channels=in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)
    
    ddim = DDIM(network=network, var_scheduler=var_scheduler)

    return ddim

def prepare_vae() -> VAE:
    """
    EXAMPLE OF INITIALIZING VAE. Feel free to change the following based on your needs and implementation.
    """
    # VAE configs
    in_channels = 1
    # NOTE: 3 down sampling layers
    mid_channels = [128, 256, 512]
    height = width = 32
    latent_dim = 256
    num_classes = 10

    # Defining the diffusion model component nets
    vae = VAE(in_channels=in_channels, 
              height=height, 
              width=width, 
              mid_channels=mid_channels, 
              latent_dim=latent_dim,
              num_classes=num_classes)
    
    return vae

def prepare_lddpm() -> LDDPM:
    """
    EXAMPLE OF INITIALIZING LDDPM. Feel free to change the following based on your needs and implementation.
    """
    # VAE configs (NOTE: it should be exactly the same config as used in prepare_vae() function)
    in_channels = 1
    mid_channels = [64, 128, 256, 512]
    height = width = 32
    latent_dim = 1
    num_classes = 10
    vae = VAE(in_channels=in_channels,
              mid_channels=mid_channels,
              height=height,
              width=width,
              latent_dim=latent_dim,
              num_classes=num_classes)
    
    # DO NOT remove the following line
    vae.load_state_dict(torch.load('checkpoints/VAE.pt'))

    # variance scheduler configs
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # Diffusion unet configs (NOTE: not more than 2 down sampling layers)
    ddpm_in_channels = latent_dim
    down_channels = [256, 512, 1024]
    up_channels = [1024, 512, 256]
    time_embed_dim = 128

    # Defining the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # TODO: defining the UNet for the diffusion model
    network = UNet(in_channels=ddpm_in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim)
    
    lddpm = LDDPM(network=network, vae=vae, var_scheduler=var_scheduler)

    return lddpm

