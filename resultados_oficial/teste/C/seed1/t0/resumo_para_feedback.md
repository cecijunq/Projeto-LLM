# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: FALHOU
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- literature_review: fase_falhou
- running_experiments: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: timeout
- running_experiments: erro_de_execucao
- running_experiments: timeout
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
1. **Dataset Preparation**: Use the CIFAR-10 dataset from torchvision, split into training and test sets.
2. **Model Selection**: Implement a simple CNN (e.g., LeNet-5) and an image transformer (e.g., a transformer with 4 attention heads and 4 layers).
3. **Noise Generation**: Add Gaussian noise to test images at different noise levels (10%, 20%, 30%).
4. **Training**: Train both models on the clean training set.
5. **Evaluation**: Evaluate both models on clean and noisy test images using accuracy as the metric.
6. **Comparison**: Compare the accuracy of both models across different noise levels to determine which model is more sensitive to pixel noise.
7. **Analysis**: Draw conclusions based on the performance comparison.
</final plan>

## Código de preparação de dados
```python
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
