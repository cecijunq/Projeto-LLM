import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

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

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, d_model=28*28, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(28*28, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Take mean over sequence
        return self.fc(x)

# Training and evaluation function
def train_and_evaluate(model, train_loader, test_loader):
    model.train()
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.view(-1, 28*28)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Training accuracy: {accuracy:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(-1, 28*28)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy

# Training and evaluation for CNN
cnn_model = CNNModel()
cnn_acc_clean = train_and_evaluate(cnn_model, train_loaders['clean'], test_loaders['clean'])
cnn_acc_10 = train_and_evaluate(cnn_model, train_loaders['10%'], test_loaders['10%'])
cnn_acc_20 = train_and_evaluate(cnn_model, train_loaders['20%'], test_loaders['20%'])
cnn_acc_30 = train_and_evaluate(cnn_model, train_loaders['30%'], test_loaders['30%'])

# Training and evaluation for Transformer
transformer_model = TransformerModel()
transformer_acc_clean = train_and_evaluate(transformer_model, train_loaders['clean'], test_loaders['clean'])
transformer_acc_10 = train_and_evaluate(transformer_model, train_loaders['10%'], test_loaders['10%'])
transformer_acc_20 = train_and_evaluate(transformer_model, train_loaders['20%'], test_loaders['20%'])
transformer_acc_30 = train_and_evaluate(transformer_model, train_loaders['30%'], test_loaders['30%'])

# Print results
print("CNN Model Accuracy:")
print(f"Clean: {cnn_acc_clean:.4f}")
print(f"10% Noise: {cnn_acc_10:.4f}")
print(f"20% Noise: {cnn_acc_20:.4f}")
print(f"30% Noise: {cnn_acc_30:.4f}")

print("\nTransformer Model Accuracy:")
print(f"Clean: {transformer_acc_clean:.4f}")
print(f"10% Noise: {transformer_acc_10:.4f}")
print(f"20% Noise: {transformer_acc_20:.4f}")
print(f"30% Noise: {transformer_acc_30:.4f}")