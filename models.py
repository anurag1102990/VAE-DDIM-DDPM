import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            # TODO: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
            
        elif interpolation == 'quadratic':
            # TODO: complete the quadratic interpolation of betas here
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps)**2
            
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        
        # TODO: add other statistics such alphas alpha_bars and all the other things you might need here
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        self.alpha_bars = self.alpha_bars.to(device)  # Move alpha_bars to the correct device

        # Flatten time_step for indexing
        time_step = time_step.to(device)

        # Generate noise
        noise = torch.randn_like(x, device = device)
        
        # Compute alpha_bar
        alpha_bar = self.alpha_bars[time_step].to(device).view(-1, *[1]*(len(x.shape)-1))
        noisy_input = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
        return noisy_input, noise

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device

        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=device))
        emb = time[:, None] * emb[None, :]  # Shape: [Batch, Half_Dim]
        embeddings = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Shape: [Batch, Dim]
        # print(f"Sinusoidal embedding output shape: {embeddings.shape}")  # Print the shape of the output embedding
        return embeddings
    
class MyBlock(nn.Module):
    def __init__(self, input_shape, input_channels, output_channels, 
                 kernel_size=3, stride=1, padding=1, use_normalization=False, 
                 activation_fn=None):
        super(MyBlock, self).__init__()

        self.group_size = 8
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Convolutional Layers
        self.conv_layer1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.conv_layer2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride, padding)

        # Group Normalization Layers
        self.norm1 = nn.GroupNorm(8, output_channels)
        self.norm2 = nn.GroupNorm(8, output_channels)

        # Activation Function
        if activation_fn is None:
            self.activation_fn = nn.SiLU() 
        else:
            activation_fn

    def forward(self, inputs):
        
        # First convolution and groupNormalisation and activation
        conv_output1 = self.conv_layer1(inputs)
        norm_output1 = self.norm1(conv_output1)
        activated_output1 = self.activation_fn(norm_output1)
        # Second convolution and groupNormalisation and activation
        conv_output2 = self.conv_layer2(activated_output1)
        norm_output2 = self.norm2(conv_output2)
        activated_output2 = self.activation_fn(norm_output2)

        # Return the value.
        return activated_output2


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, 
                 down_channels: List = [64, 128, 128, 128],
                 up_channels: List = [128, 128, 128, 64],
                 time_emb_dim: int = 128,
                 num_classes: int = 10) -> None:
        super(UNet, self).__init__()

        self.num_classes = num_classes

        # Embedding Layers
        self.temporal_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        # print(time_emb_dim.device)
        self.label_embedding = nn.Embedding(num_classes, time_emb_dim)

        # 4 layer Encoder Path
        self.time_emb1 = self._build_time_embedding(time_emb_dim, in_channels)
        self.time_emb2 = self._build_time_embedding(time_emb_dim, down_channels[0])
        self.time_emb3 = self._build_time_embedding(time_emb_dim, down_channels[1])
        self.time_emb4 = self._build_time_embedding(time_emb_dim, down_channels[2])
        
        self.mid_time_emb = self._build_time_embedding(time_emb_dim, down_channels[3])
        
        self.time_emb5 = self._build_time_embedding(time_emb_dim, up_channels[0])
        self.time_emb6 = self._build_time_embedding(time_emb_dim, up_channels[1])
        self.time_emb7 = self._build_time_embedding(time_emb_dim, up_channels[2])
        self.time_emb8 = self._build_time_embedding(time_emb_dim, up_channels[3])


        self.encoder_block1 = MyBlock((in_channels, 32, 32), 
                                      in_channels, 
                                      down_channels[0])
        
        self.encoder_block2 = MyBlock((down_channels[0], 16, 16), 
                                      down_channels[0], 
                                      down_channels[1])
        
        self.encoder_block3 = MyBlock((down_channels[1], 8, 8), 
                                      down_channels[1], 
                                      down_channels[2])
    
        self.encoder_block4 = MyBlock((down_channels[2], 4, 4), 
                                      down_channels[2], 
                                      down_channels[3])

        self.decoder_block1 = MyBlock((up_channels[0] * 2, 4, 4), 
                                      up_channels[0] + down_channels[3], 
                                      up_channels[0])
        
        self.decoder_block2 = MyBlock((up_channels[1] * 2, 8, 8), 
                                      up_channels[1] + down_channels[2], 
                                      up_channels[1])

        self.decoder_block3 = MyBlock((up_channels[2] * 2, 16, 16), 
                                      up_channels[2] + down_channels[1], 
                                      up_channels[2])

        self.decoder_block4 = MyBlock((up_channels[3] * 2, 32, 32), 
                                      up_channels[3] + down_channels[0], 
                                      up_channels[3])
        
        self.downsample1 = nn.Conv2d(down_channels[0], 
                                     down_channels[0], 
                                     kernel_size=4, 
                                     stride=2, 
                                     padding=1)

        self.downsample2 = nn.Conv2d(down_channels[1], 
                                     down_channels[1], 
                                     kernel_size=4, 
                                     stride=2, 
                                     padding=1)


        self.downsample3 = nn.Conv2d(down_channels[2], 
                                     down_channels[2], 
                                     kernel_size=4, 
                                     stride=2, 
                                     padding=1)


        self.downsample4 = nn.Conv2d(down_channels[3], 
                                     down_channels[3], 
                                     kernel_size=4, 
                                     stride=2, 
                                     padding=1)

        # Bottleneck
        self.bottleneck_block = MyBlock((down_channels[3], 2, 2), 
                                        down_channels[3], 
                                        128)
        self.upconv1 = nn.ConvTranspose2d(128, 
                                          up_channels[0], 
                                          kernel_size=4, 
                                          stride=2, 
                                          padding=1)

        self.upconv2 = nn.ConvTranspose2d(up_channels[0], 
                                          up_channels[1], 
                                          kernel_size=4, 
                                          stride=2, 
                                          padding=1)

        self.upconv3 = nn.ConvTranspose2d(up_channels[1], 
                                          up_channels[2], 
                                          kernel_size=4, 
                                          stride=2, 
                                          padding=1)

        self.upconv4 = nn.ConvTranspose2d(up_channels[2], 
                                          up_channels[3], 
                                          kernel_size=4, 
                                          stride=2, 
                                          padding=1)


        self.final_conv = nn.Conv2d(up_channels[3], 1, kernel_size=1)

    def _build_time_embedding(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Embedding computation
        t = self.temporal_embedding(timestep)
        l = self.label_embedding(label)
        combined_emb = t + l
        batch = x.size(0)

        # Time Embeddings for Downsampling
        encoded1 = self.encoder_block1(x + (self.time_emb1(combined_emb).view(batch, -1, 1, 1)))
        down1 = self.downsample1(encoded1)

        encoded2 = self.encoder_block2(down1 + (self.time_emb2(combined_emb).view(batch, -1, 1, 1)))
        down2 = self.downsample2(encoded2)

        encoded3 = self.encoder_block3(down2 + (self.time_emb3(combined_emb).view(batch, -1, 1, 1)))
        down3 = self.downsample3(encoded3)

        encoded4 = self.encoder_block4(down3 + (self.time_emb4(combined_emb).view(batch, -1, 1, 1)))
        bottleneck_input = self.downsample4(encoded4)

        # Bottleneck
        mid_emb = self.mid_time_emb(combined_emb).view(batch, -1, 1, 1)
        bottleneck = self.bottleneck_block(bottleneck_input + mid_emb)

        # Time Embeddings for Upsampling
        up1 = self.upconv1(bottleneck)
        up1 = torch.cat([up1 + (self.time_emb5(combined_emb).view(batch, -1, 1, 1)), encoded4], dim=1)
        decoded1 = self.decoder_block1(up1)

        up2 = self.upconv2(decoded1)
        up2 = torch.cat([up2 + (self.time_emb6(combined_emb).view(batch, -1, 1, 1)), encoded3], dim=1)
        decoded2 = self.decoder_block2(up2)

        up3 = self.upconv3(decoded2)
        up3 = torch.cat([up3 + (self.time_emb7(combined_emb).view(batch, -1, 1, 1)), encoded2], dim=1)
        decoded3 = self.decoder_block3(up3)

        up4 = self.upconv4(decoded3)
        up4 = torch.cat([up4 + (self.time_emb8(combined_emb).view(batch, -1, 1, 1)), encoded1], dim=1)
        decoded4 = self.decoder_block4(up4)

        # Final layer
        return self.final_conv(decoded4)

class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int = 32, 
                 width: int = 32, 
                 mid_channels: List[int] = [128, 256, 512], 
                 latent_dim: int = 32, 
                 num_classes: int = 10) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.width = width
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        # Class label embedding
        self.class_emb = nn.Embedding(num_classes, latent_dim)
        self.height = height

        # Define encoder layers
        encoder_blocks = []
        prev_channels = in_channels
        
        for next_channels in mid_channels:
            encoder_blocks.extend([
                nn.Conv2d(prev_channels, next_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ])
            prev_channels = next_channels
        self.encoder = nn.Sequential(*encoder_blocks)

        # Dynamically calculate bottleneck size and flattened dimension
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.height, self.width)
            encoded_output_shape = self.encoder(sample_input).shape
            self.bottleneck_size = list(encoded_output_shape[1:])
            self.flatten_dim = int(torch.prod(torch.tensor(self.bottleneck_size)))
        # print("Bottleneck size:", self.bottleneck_size)
        # print("Flattened dimension:", self.flatten_dim)

        # Mean and log variance computation layers
        # Latent projection layer
        self.latent_proj2 = nn.Linear(512 * 17, self.flatten_dim)
        self.mean_layer = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, latent_dim)
        # Latent projection layer
        self.latent_proj1 = nn.Linear(512, 512 * 17)



        # Decoder layers
        decoder_blocks = []
        intermediate_channels = mid_channels[-1]
        
        for out_channels in reversed(mid_channels):
            decoder_blocks.extend([
                nn.ConvTranspose2d(
                    intermediate_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            intermediate_channels = out_channels

        decoder_blocks.append(nn.ConvTranspose2d(intermediate_channels, in_channels, kernel_size=3, stride=1, padding=1))
        decoder_blocks.append(nn.Sigmoid())  # For binary cross-entropy
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode the input
        encoded_features = self.encoder(x).flatten(start_dim=1)
        # print("Encoded feature shape:", encoded_features.shape)

        # Compute mean and log variance
        latent_mean = self.mean_layer(encoded_features)
        latent_logvar = self.logvar_layer(encoded_features)

        # Reparameterize to sample latent vector
        latent_sample = self.reparameterize(latent_mean, latent_logvar)

        # Decode latent vector
        reconstructed = self.decode(latent_sample, label)
        return reconstructed, latent_mean, latent_logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick
        std_dev = torch.exp(0.5 * logvar)
        random_noise = torch.randn_like(std_dev)
        return mean + random_noise * std_dev

    def decode(self, latent_vector: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Embed label and concatenate with latent vector
        label_embedding = self.class_emb(label)
        # print(label_embedding)
        latent_with_label = torch.cat([latent_vector, label_embedding], dim=1)

        # Project latent space to bottleneck size
        proj_layer1 = F.relu(self.latent_proj1(latent_with_label))
        proj_layer2 = F.relu(self.latent_proj2(proj_layer1))
        reshaped_latent = proj_layer2.view( -1, * self.bottleneck_size) # Reshape to match bottleneck dimensions
        # print(reshaped_latent)

        # Decode reshaped bottleneck representation
        return self.decoder(reshaped_latent)

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor = None):
        # Generate random labels if none provided
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)

        # Sample latent space with random noise
        latent_noise = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode the generated latent representation
        return self.decode(latent_noise, labels)

    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(pred, target, reduction='sum')

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


# class LDDPM(nn.Module):
#     def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
#         super().__init__()

#         self.var_scheduler = var_scheduler
#         self.vae = vae
#         self.network = network

#         # freeze vae
#         self.vae.requires_grad_(False)
    
#     def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         # TODO: uniformly sample as many timesteps as the batch size
#         _, mu, logvar = self.vae(x, label)
#         z = self.vae.reparameterize(mu, logvar)

#         # Randomly sample timesteps for diffusion
#         t = torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)
        
#         # TODO: generate the noisy input
#         noisy_z, noise = self.var_scheduler.add_noise(z, t)

#         # TODO: estimate the noise
#         estimated_noise = self.network(noisy_z, t, label)

#         # compute the loss (either L1 or L2 loss)
#         loss = F.mse_loss(estimated_noise, noise)
#         return loss

#     @torch.no_grad()
#     def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
#         # TODO: implement the sample recovery strategy of the DDPM
#         sample = ...

#         return sample

#     @torch.no_grad()
#     def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
#         if labels is not None:
#             assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
#             labels = labels.to(device)
#         else:
#             labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
#         # TODO: using the diffusion model generate a sample inside the latent space of the vae
#         # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
#         z = torch.randn(num_samples, self.vae.latent_dim, device=device)
        
#         for t in reversed(range(self.var_scheduler.num_steps)):
#             estimated_noise = self.network(z, t, labels)
#             z = self.var_scheduler.recover_sample(z, estimated_noise, t)

#         sample = self.vae.decode(sample, labels)
#         return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()
        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)

        # Generate noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # Debugging
        # print(f"x: {x.shape}, noisy_input: {noisy_input.shape}, noise: {noise.shape}")

        # Estimate noise
        estimated_noise = self.network(noisy_input, t, label)

        # Debugging
        # print(f"estimated_noise: {estimated_noise.shape}")

        # Compute loss
        loss = F.l1_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Move variance scheduler tensors to the same device as `noisy_sample`
        device = noisy_sample.device
        self.var_scheduler.alphas = self.var_scheduler.alphas.to(device)
        self.var_scheduler.alpha_bars = self.var_scheduler.alpha_bars.to(device)
        self.var_scheduler.betas = self.var_scheduler.betas.to(device)

        # print(f"noisy_sample device: {noisy_sample.device}, alphas device: {self.var_scheduler.alphas.device}")

        # Retrieve parameters for the current timestep
        alpha_t = self.var_scheduler.alphas[timestep][:, None, None, None]
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep][:, None, None, None]
        beta_t = self.var_scheduler.betas[timestep][:, None, None, None]

        # Compute the mean for x_{t-1}
        mean = (noisy_sample - beta_t * estimated_noise / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)

        # Add stochastic noise for sampling if t > 1
        noise = torch.randn_like(noisy_sample) if timestep[0] > 1 else 0
        sample = mean + torch.sqrt(beta_t) * noise

        return sample


    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples], device=device)
        else:
            labels = None

        # Initialize the noise sample
        sample = torch.randn(num_samples, 1, 32, 32, device=device)

        # Iterate over timesteps in reverse
        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)  # Create a batch of timesteps
            estimated_noise = self.network(sample, t_tensor, labels)  # Use batch timesteps
            sample = self.recover_sample(sample, estimated_noise, t_tensor)

        return sample



class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss
        loss = F.mse_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_input: torch.Tensor, predicted_noise: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        # Apply the DDIM sample recovery strategy
        device = step.device

        # Transfer scheduler parameters to the device
        alpha_values = self.var_scheduler.alphas.to(device)
        alpha_cumprod = self.var_scheduler.alpha_bars.to(device)
        alpha_bar_curr = alpha_cumprod[step[0]]
        alpha_bar_prev = alpha_cumprod[step[0] - 1] if step[0] > 0 else torch.tensor(1.0, device=device)
        sample_t = noisy_input

        recovered_sample = (
            (alpha_bar_prev.sqrt() / alpha_bar_curr.sqrt()) * (sample_t - (1 - alpha_bar_curr).sqrt() * predicted_noise)
            + (1 - alpha_bar_prev).sqrt() * predicted_noise
        )

        return recovered_sample

    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples], device=device)
        else:
            labels = None

        # Initialize the noise sample
        sample = torch.randn(num_samples, 1, 32, 32, device=device)

        # Iterate over timesteps in reverse
        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)  # Create a batch of timesteps
            estimated_noise = self.network(sample, t_tensor, labels)  # Use batch timesteps
            sample = self.recover_sample(sample, estimated_noise, t_tensor)

        return sample  
