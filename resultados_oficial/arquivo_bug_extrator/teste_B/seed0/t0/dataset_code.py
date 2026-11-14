from datasets import load_dataset
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16

# Load the dataset
dataset = load_dataset('uoft-cs/cifar10', 'plain_text', split='test[:500]')

# Convert dataset to PyTorch format
images = [np.array(img) for img in dataset['img']]
labels = dataset['label']

# Normalize images to [0, 1]
images = [img / 255.0 for img in images]

# Convert to tensors
images_tensor = torch.tensor(images).float()
labels_tensor = torch.tensor(labels).long()

# Function to add random pixel noise
def add_random_pixel_noise(image_tensor, noise_level):
    noise = np.random.uniform(-noise_level, noise_level, image_tensor.shape)
    noisy_image = image_tensor + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image