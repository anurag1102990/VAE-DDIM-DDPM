import torch
import time
import warnings

import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models import VAE
from models import DDPM
from models import DDIM
from models import LDDPM

warnings.filterwarnings('ignore')

def train_ddpm_ddim(diffusion_model: DDPM | DDIM, 
                    train_loader: DataLoader, 
                    epochs: int, 
                    lr: float=2e-4, 
                    device=torch.device('cuda'), 
                    eval_intervals: int=10, 
                    num_samples: int=5):
    
    torch.cuda.empty_cache()
    model_name = 'DDPM' if isinstance(diffusion_model, DDPM) else 'DDIM'

    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
  
    for epoch in range(epochs):
        avg_loss = 0.
        start = time.time()
        for sample in train_loader:
            x, label = sample[0].to(device), sample[1].to(device)
            
            optimizer.zero_grad()

            loss = diffusion_model(x, label)
            
            loss.backward()
            
            try:
                clip_grad_norm_(diffusion_model.parameters(), 1.)
            except:
                pass
            
            optimizer.step()

            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        
        # validating and saving the model
        print(f'Epoch: {epoch} - Loss: {avg_loss:.3f} ({time.time() - start:.2f} sec)')
        lr_scheduler.step(avg_loss)

        if epoch % eval_intervals == 0 or epoch == epochs - 1:
            # generate some sample to see the quality of the generative model
            with torch.no_grad():
                samples = diffusion_model.generate_sample(num_samples, torch.device('cuda'))

            fig, ax = plt.subplots(1, num_samples)
            fig.set_size_inches(3 * num_samples, 10)
            for i in range(num_samples):
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
            fig.savefig(f'{model_name}_sample.png')
            plt.close()
            torch.save(diffusion_model.state_dict(), f'checkpoints/{model_name}.pt')

def train_vae(vae: VAE, 
              train_loader: DataLoader, 
              epochs: int, 
              lr: float=3e-4, 
              device=torch.device('cuda'), 
              num_samples: int=5,
              eval_intervals: int=5):
    
    torch.cuda.empty_cache()
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        avg_kl_loss = 0.
        avg_recon_loss = 0.
        avg_loss = 0.
        start = time.time()
        for sample in train_loader:
            x, label = sample[0].to(device), sample[1].to(device)

            optimizer.zero_grad()

            out, mean, logvar = vae(x, label)

            recon_loss = vae.reconstruction_loss(out, x)
            kl_loss = vae.kl_loss(mean, logvar)

            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            avg_recon_loss += recon_loss.item()
            avg_kl_loss += kl_loss.item()
            avg_loss += loss.item()
        
        avg_recon_loss /= len(train_loader)
        avg_kl_loss /= len(train_loader)
        avg_loss /= len(train_loader)

        print(f'Epoch: {epoch} - Recon Loss: {avg_recon_loss:.3f} - KL Loss: {avg_kl_loss:.3f} - Loss: {avg_loss:.3f} ({time.time() - start:.2f}sec)')

        if epoch % eval_intervals == 0 or epoch == epochs - 1:
            samples = vae.generate_sample(num_samples, device)
            fig, ax = plt.subplots(1, num_samples)
            fig.set_size_inches(3 * num_samples, 10)
            for i in range(num_samples):
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
            fig.savefig(f'VAE_sample.png')
            plt.close()
            torch.save(vae.state_dict(), f'checkpoints/VAE.pt')

def train_lddpm(diffusion_model: LDDPM, 
                train_loader: DataLoader, 
                epochs: int, 
                lr: float=2e-4,
                device=torch.device('cuda'),
                eval_intervals: int=10,
                num_samples: int=5):
  
  torch.cuda.empty_cache()
  optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
  
  for epoch in range(epochs):
        avg_loss = 0.
        start = time.time()
        for sample in train_loader:
            x, label = sample[0].to(device), sample[1].to(device)
            
            optimizer.zero_grad()

            # encode image
            with torch.no_grad():
                _, mu, logvar = diffusion_model.vae(x)
                # sample from the latent distribution
                x = diffusion_model.vae.reparameterize(mu, logvar)
            
            loss = diffusion_model(x, label)
            
            loss.backward()
            
            try:
                clip_grad_norm_(diffusion_model.parameters(), 1.)
            except:
                pass
            
            optimizer.step()

            avg_loss += loss.item()

        avg_loss /= len(train_loader)

        # validating and saving the model
        print(f'Epoch: {epoch} - Loss: {avg_loss:.3f} ({time.time() - start:.2f}sec)')
        lr_scheduler.step(avg_loss)

        if epoch % eval_intervals == 0 or epoch == epochs - 1:
            # generate some sample to see the quality of the generative model
            with torch.no_grad():
                samples = diffusion_model.generate_sample(num_samples, torch.device('cuda'), clamp=False)
            fig, ax = plt.subplots(1, num_samples)
            fig.set_size_inches(3 * num_samples, 10)
            for i in range(5):
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
            fig.savefig(f'LDDPM_sample.png')
            plt.close()
            torch.save(diffusion_model.state_dict(), f'checkpoints/LDDPM.pt')
