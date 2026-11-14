import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the HF_TOKEN environment variable (this is just an example; you need to set it in your environment)
os.environ["HF_TOKEN"] = "your_hf_token_here"

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert datasets to numpy arrays
train_data = train_dataset.data.numpy() / 255.0
test_data = test_dataset.data.numpy() / 255.0
train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()

# Define noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Function to add noise to images
def add_noise(images, noise_level):
    noise = np.random.normal(0, noise_level, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images

# Create noisy training datasets
noisy_train_datasets = []
for noise_level in noise_levels:
    noisy_train_data = add_noise(train_data, noise_level)
    noisy_train_dataset = TensorDataset(torch.from_numpy(noisy_train_data), torch.from_numpy(train_labels))
    noisy_train_datasets.append(noisy_train_dataset)

# Define data loaders
batch_size = 64
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in noisy_train_datasets]
test_loader = DataLoader(TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels)), batch_size=batch_size, shuffle=False)