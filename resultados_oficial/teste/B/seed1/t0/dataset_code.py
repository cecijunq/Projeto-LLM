import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split the dataset into train and test
train_data = train_dataset.data[:5000].float() / 255.0
train_labels = train_dataset.targets[:5000]

test_data = test_dataset.data[:5000].float() / 255.0
test_labels = test_dataset.targets[:5000]

# Convert to tensors
train_data = train_data.view(-1, 28*28)
test_data = test_data.view(-1, 28*28)

# Function to add noise
def add_noise(data, noise_level):
    noise = torch.randn(data.shape) * noise_level
    noisy_data = data + noise
    noisy_data = torch.clamp(noisy_data, 0, 1)
    return noisy_data

# Add noise to data
train_data_noisy_10 = add_noise(train_data, 0.1)
train_data_noisy_20 = add_noise(train_data, 0.2)
train_data_noisy_30 = add_noise(train_data, 0.3)

test_data_noisy_10 = add_noise(test_data, 0.1)
test_data_noisy_20 = add_noise(test_data, 0.2)
test_data_noisy_30 = add_noise(test_data, 0.3)

# Create datasets
train_datasets = {
    'clean': TensorDataset(train_data, train_labels),
    '10%': TensorDataset(train_data_noisy_10, train_labels),
    '20%': TensorDataset(train_data_noisy_20, train_labels),
    '30%': TensorDataset(train_data_noisy_30, train_labels)
}

test_datasets = {
    'clean': TensorDataset(test_data, test_labels),
    '10%': TensorDataset(test_data_noisy_10, test_labels),
    '20%': TensorDataset(test_data_noisy_20, test_labels),
    '30%': TensorDataset(test_data_noisy_30, test_labels)
}

# Create data loaders
train_loaders = {k: DataLoader(v, batch_size=64, shuffle=True) for k, v in train_datasets.items()}
test_loaders = {k: DataLoader(v, batch_size=64, shuffle=False) for k, v in test_datasets.items()}