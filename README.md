# Generative Models with DDPM, DDIM, and VAE – Class-Conditional Image Generation

This repository contains my implementation of various **generative models** as part of CMPUT 328 (Visual Recognition). The goal was to train models like **DDPM**, **DDIM**, and **VAE** on the **FashionMNIST** dataset to generate realistic images conditioned on specific classes.

## 🧠 Project Overview

Generative models aim to learn data distributions and produce realistic samples. This project focuses on **class-conditional generation**, where the model learns to generate images belonging to a specific class (e.g., shirts, bags, shoes from FashionMNIST).

The models implemented include:

- **DDPM (Denoising Diffusion Probabilistic Models)**  
- **DDIM (Denoising Diffusion Implicit Models)**  
- **Conditional Variational Autoencoders (VAE)**  

All models are trained on a **32×32 padded version of FashionMNIST**, which aids in better downsampling and convolution operations.

---

## 📦 Implemented Models

### 1. DDPM
- Learns to iteratively **denoise** pure noise over 1000 time steps.
- Conditioned on class labels using sinusoidal time embeddings + label embeddings.
- Key features:
  - Custom UNet for noise prediction.
  - Quadratic beta scheduling for variance.
  - Implements `recover_sample()` and `generate_sample()` functions.

### 2. DDIM
- A **deterministic variant** of DDPM.
- Shares the same architecture and scheduler, but applies a different sampling method in `recover_sample()`.

### 3. Conditional VAE
- Encodes input images into latent vectors with class information.
- Uses reparameterization trick and binary cross-entropy loss.
- Supports sampling via `generate_sample()` using random latent vectors.

---

## 🗂️ Files and Structure

- `models.py` – UNet, VAE, DDPM, DDIM, and LDDPM implementations.
- `assignment.py` – Model builders for training/evaluation.
- `main.py` – Training and inference script.
- `classifier.pt` – Pretrained model used to evaluate accuracy on generated samples.

---

## 📊 Results

The generated images were evaluated using a pretrained classifier. Below are the key outcomes:

| Model   | Accuracy (Generated Samples) |
|---------|------------------------------|
| **DDPM**   | 86.2%                         |
| **DDIM**   | 84.5%                         |
| **VAE**    | 83.1%                         |

> 🏆 Models scoring above 85% were considered high-performing per grading rubric.

---

## 🧪 How to Run

### 🔧 Train a Model
```bash
python main.py --model ddpm --mode train
python main.py --model vae --mode train
python main.py --model ddpm --mode generate
python main.py --model vae --mode generate
```

## Dataset
- FashionMNIST: 10-class dataset of grayscale clothing images.
- Images are padded to 32×32 for smoother convolutions and architecture compatibility.
- Used in class-conditional format: model receives a class label and generates corresponding images.

