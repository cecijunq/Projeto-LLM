# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- literature_review: formato_invalido
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2112.09300v1] This paper proposes an end-to-end image compression and analysis model using Transformers, aiming to improve cloud-based image classification. The model integrates a lightweight convolutional neural network with a Transformer to handle compressed features for classification. While the paper focuses on compression and analysis, it highlights the potential of Transformers in processing image data, suggesting their capability to handle noisy or compressed features. However, it does not directly compare noise sensitivity between Transformers and convolutional networks.
[2111.10480v6] TransMorph: Transformer for unsupervised medical image registration introduces a hybrid Transformer-ConvNet model for volumetric medical image registration. The paper highlights the potential of Transformers in capturing long-range spatial relationships, which is crucial for image registration tasks. While the paper does not directly compare noise sensitivity between Transformers and convolutional networks, it suggests that Transformers may offer superior spatial understanding, potentially leading to better robustness in the presence of noise. However, noise sensitivity analysis is not explicitly addressed in this work.
[2402.16221v1] This paper explores the integration of preprocessing techniques with convolutional neural networks (CNNs) for tumor detection in medical imaging. While the primary focus is on enhancing image features through preprocessing methods like Gaussian smoothing and K-means clustering, the paper indirectly highlights the robustness of CNNs in handling noisy or preprocessed medical images. It does not explicitly compare noise sensitivity between CNNs and image transformers, but the emphasis on preprocessing suggests that CNNs are designed to be more resilient to noise in the input data. This could imply that CNNs are more noise-sensitive than image transformers in certain scenarios, though this is not directly evaluated in the paper.
[2205.04204v2] This paper proposes a residual Swin-Transformer based regularizer (RSTR) for PET image reconstruction, addressing the challenges of noise in PET data. The study highlights the potential of Transformers in handling noisy data by leveraging their global attention mechanisms, which can capture long-range dependencies and improve reconstruction quality. While the paper primarily focuses on PET image reconstruction, it indirectly addresses noise sensitivity by comparing the performance of Transformers with traditional convolutional networks. The results suggest that Transformers can offer better robustness to noise in certain scenarios, although the paper does not directly evaluate noise sensitivity between Transformers and convolutional networks in image processing tasks.
[2205.04204v2] This paper proposes a residual Swin-Transformer based regularizer (RSTR) for PET image reconstruction, addressing the challenges of noise in PET data. The study highlights the potential of Transformers in handling noisy data by leveraging their global attention mechanisms, which can capture long-range dependencies and improve reconstruction quality. While the paper primarily focuses on PET image reconstruction, it indirectly addresses noise sensitivity by comparing the performance of Transformers with traditional convolutional networks. The results suggest that Transformers can offer better robustness to noise in certain scenarios, although the paper does not directly evaluate noise sensitivity between Transformers and convolutional networks in image processing tasks.

## Plano
We will design a simple experiment to compare the noise sensitivity of image transformers and convolutional networks. We will use the MNIST dataset and add varying levels of Gaussian noise to the images. We will train a CNN and a Vision Transformer (ViT) on the noisy images and evaluate their performance on the original test set. We will repeat the experiment for multiple noise levels to observe how each model's accuracy changes with increasing noise. This will allow us to compare the robustness of each model to pixel noise.
</final plan>

## Código de preparação de dados
```python
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
```

## Código do experimento
```python

```

## Saída do experimento
```
Nenhum código executou com sucesso.
```

## Interpretação
(sem interpretação)

## Relatório final

(sem relatório)
