# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- literature_review: busca_em_loop
- data_preparation: dependencia_ausente
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (3)
[2112.09300v1] This paper proposes an end-to-end image compression and analysis model using Transformers, aiming to improve image classification from compressed features. The study replaces the patchify stem of the Vision Transformer (ViT) with a lightweight convolutional neural network (CNN) to inject convolutional inductive bias. While the paper focuses on compression and classification, it highlights the integration of CNN-based preprocessing with Transformer models, which may provide insights into the noise sensitivity of Transformers compared to CNNs in image processing tasks.
[2111.10480v6] This paper, titled "TransMorph: Transformer for unsupervised medical image registration," explores the use of Transformers in medical imaging tasks, specifically image registration. The study highlights that Transformers can effectively capture long-range spatial relationships, which are often challenging for ConvNets. The paper presents TransMorph, a hybrid model combining Transformers and ConvNets, which shows promising results in volumetric medical image registration. This work provides insights into the potential advantages of Transformers over ConvNets in medical imaging applications, especially in handling noise and capturing spatial relationships.
[2201.09873v1] This paper, titled "Transformers in Medical Imaging: A Survey," provides a comprehensive review of the applications of Transformers in medical imaging. The paper highlights that Transformers have shown significant promise in capturing global context, which is often challenging for ConvNets with local receptive fields. While the paper does not directly address noise sensitivity, it sets the stage for understanding the broader context of Transformer applications in medical imaging, which can inform future research into their robustness to noise. The survey also discusses various architectural designs and challenges, which may indirectly relate to noise sensitivity in medical imaging tasks.

## Plano
In this experiment, we will compare the sensitivity of image transformers and convolutional networks to pixel noise by evaluating their performance on the MNIST dataset under varying levels of Gaussian noise. The experiment will follow these steps:

1. **Data Preparation**: Load the MNIST dataset and add varying levels of Gaussian noise to the images. We will use noise levels of 0% (clean), 10%, 20%, 30%, and 40% to create five different noise conditions.

2. **Model Selection**: Use a lightweight CNN (e.g., a simple CNN with two convolutional layers) and a Vision Transformer (ViT) model. Both models will be trained on the clean MNIST dataset.

3. **Evaluation**: For each noise level, evaluate both models on the noisy images and measure their accuracy. This will allow us to compare how well each model performs under different noise conditions.

4. **Analysis**: Compare the accuracy of the CNN and ViT across all noise levels. The goal is to determine whether image transformers are more or less sensitive to pixel noise than convolutional networks.

## Código de preparação de dados
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = load_digits()
train_data, test_data, train_labels, test_labels = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Generate synthetic noisy digits data
def add_gaussian_noise(images, noise_level):
    noise = np.random.normal(0, noise_level, images.shape)
    noisy_image = images + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

# Define noise levels
noise_levels = [0, 0.1, 0.2, 0.3, 0.4]

# Create a DataFrame to store the data
data = []

# Add clean and noisy data
for noise_level in noise_levels:
    noisy_data = add_gaussian_noise(train_data, noise_level)
    for i in range(len(train_data)):
        data.append({
            'image': noisy_data[i].flatten(),  # Flatten image for easier processing
            'label': train_labels[i]
        })

# Convert to DataFrame
df = pd.DataFrame(data)
print("Columns:", df.columns)
print("Example:", df.iloc[0])
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
