import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score
import os
import math

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Load CIFAR-10 dataset from Hugging Face
from datasets import load_dataset

dataset = load_dataset('uoft-cs/cifar10', 'plain_text')

# Split the dataset into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Convert to PyTorch tensors
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

# Split the training set into training and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)