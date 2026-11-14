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
- data_preparation: erro_de_execucao
- running_experiments: outro_erro
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: dependencia_ausente
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
1. **Dataset**: Use CIFAR-10 from torchvision for image classification.
2. **Models**: Train a CNN (e.g., ResNet-18) and a Vision Transformer (ViT) on the clean CIFAR-10 dataset.
3. **Noise Generation**: Add random pixel noise to images (e.g., 5%, 10%, 15%, 20% noise levels) using a uniform distribution.
4. **Evaluation**: Test the models on noisy images and record accuracy for each noise level.
5. **Statistical Analysis**: Compute confidence intervals for accuracy differences between models at each noise level.
6. **Comparison**: Analyze results to determine which model is more robust to pixel noise.
7. **Tools**: Use Ollama to serve the local language model for documentation and analysis (no paid APIs, no model downloads).
</final plan>

## Código de preparação de dados
```python
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
