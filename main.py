import os
import torch
import argparse
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets.mnist import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Pad
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader
from classifier import Classifier
from classifier import calc_acc
from assignment import prepare_ddpm
from assignment import prepare_ddim
from assignment import prepare_vae
from assignment import prepare_lddpm
from trainers import train_ddpm_ddim
from trainers import train_vae
from trainers import train_lddpm

warnings.filterwarnings('ignore')


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

def load_data(normalize: bool=True):
    if normalize:
        train_set = FashionMNIST('FashionMNIST', train=True, download=True, 
                                transform=Compose([ToTensor(), Pad([2, 2, 2, 2]), Lambda(lambda x: 2 * x - 1.)]))
    else:
        train_set = FashionMNIST('FashionMNIST', train=True, download=True, 
                                 transform=Compose([ToTensor(), Pad([2, 2, 2, 2])]))
    
    return train_set

if __name__ == '__main__':
    SEED = 406
    set_seeds(SEED) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ddpm')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    model_type = args.model
    mode = args.mode

    assert model_type in ['ddpm', 'ddim', 'vae', 'lddpm'], 'Wrong model type selected! Only ddpm and ddim are supported!'
    assert mode in ['train', 'generate']

    
    # specifying the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('[!] WARNING, training on CPU could take long!')
        device = torch.device('cpu')

    # specifying the model
    if model_type == 'ddpm':
        model = prepare_ddpm()
        trainer = train_ddpm_ddim
    elif model_type == 'ddim':
        model = prepare_ddim()
        trainer = train_ddpm_ddim
    elif model_type == 'vae':
        model = prepare_vae()
        trainer = train_vae
    else:
        model = prepare_lddpm()
        trainer = train_lddpm

    model = model.to(device)

    if mode == 'train':
        os.makedirs('checkpoints', exist_ok=True)
        # training configs
        epochs = 39
        batch_size = 128
        eval_intervals = 10
        num_generated_samples = 5
        lr = 1e-4

        # loading the dataset
        normalize = True if model_type in ['ddpm', 'ddim'] else False
        train_loader = DataLoader(load_data(normalize), batch_size=batch_size, shuffle=True)
        print(f'\n\n\t Training {model_type.upper()} ...')
        trainer(model, train_loader, 
                epochs=epochs, 
                lr=lr, 
                device=device, 
                eval_intervals=eval_intervals, 
                num_samples=num_generated_samples)
    else:
        if os.path.exists(f'checkpoints/{model_type.upper()}.pt'):
            model.load_state_dict(torch.load(f'checkpoints/{model_type.upper()}.pt'))
        else:
            raise FileNotFoundError(f'[!] ERROR: Weight file for {model_type.upper()} does not exist! Try training the model first.')
        
        print(f'\n\n\t Generating images using {model_type.upper()} ...')
        with torch.no_grad():
            labels = torch.tensor([i//5 for i in range(50)], dtype=torch.long, device=device)
            samples = model.generate_sample(50, device=device, labels=labels)

            fig, ax = plt.subplots(10, 5)
            fig.set_size_inches(20, 40)
            for i in range(10):
                for j in range(5):
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    ax[i, j].imshow(samples[i * 5 + j].cpu().permute(1, 2, 0).numpy(), cmap='gray')
            fig.savefig(f'{model_type.upper()}_generated_samples.png')
            plt.close()

            classifier = Classifier().to(device)
            classifier.load_state_dict(torch.load('classifier.pt'))

            predictions = classifier(samples)

            acc, score = calc_acc(predictions, labels)
        
        print(f'\n\n\t Generated images saved!')
        print(f'Accuracy: {acc * 100:.2f}%\t Score: {score:.2f}')


    
