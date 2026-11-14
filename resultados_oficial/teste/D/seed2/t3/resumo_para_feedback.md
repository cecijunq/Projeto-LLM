# Projeto: Does rotation data augmentation make a small CNN more robust to rotated images on CIFAR-10?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- running_experiments: erro_de_execucao
- running_experiments: erro_de_sintaxe
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2310.04414v3] The paper introduces CIFAR-10-Warehouse, a testbed consisting of 180 datasets that provide a broad and more realistic analysis of model generalization. These datasets are collected through various prompting methods on image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This testbed is particularly relevant for evaluating how well models generalize to out-of-distribution data, such as rotated images, which is crucial for understanding the robustness of CNNs under data augmentation techniques like rotation.
[2411.15497v3] The paper proposes AeroGen, a layout-controllable diffusion generative model tailored for remote sensing image object detection (RSIOD). This model addresses the scarcity of labeled data in RSIOD datasets by enabling the generation of high-quality images with horizontal and rotated bounding box conditions. AeroGen is the first model to support both types of bounding box condition generation, which is crucial for enhancing the performance of object detection algorithms in remote sensing scenarios. Although the paper focuses on remote sensing, the techniques for generating rotated images and the use of diffusion models are relevant to the study of rotation data augmentation and its impact on CNN robustness in datasets like CIFAR-10.
[2310.04414v3] The paper introduces CIFAR-10-Warehouse, a testbed consisting of 180 datasets that provide a broad and more realistic analysis of model generalization. These datasets are collected through various prompting methods on image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This testbed is particularly relevant for evaluating how well models generalize to out-of-distribution data, such as rotated images, which is crucial for understanding the robustness of CNNs under data augmentation techniques like rotation.
[2310.04414v3] The paper introduces CIFAR-10-Warehouse, a testbed consisting of 180 datasets that provide a broad and more realistic analysis of model generalization. These datasets are collected through various prompting methods on image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This testbed is particularly relevant for evaluating how well models generalize to out-of-distribution data, such as rotated images, which is crucial for understanding the robustness of CNNs under data augmentation techniques like rotation.
[2310.04414v3] The paper introduces CIFAR-10-Warehouse, a testbed consisting of 180 datasets that provide a broad and more realistic analysis of model generalization. These datasets are collected through various prompting methods on image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This testbed is particularly relevant for evaluating how well models generalize to out-of-distribution data, such as rotated images, which is crucial for understanding the robustness of CNNs under data augmentation techniques like rotation.

## Plano
Evaluate the robustness of a small CNN (e.g., VGG-11) to rotated images on CIFAR-10 by comparing its performance with and without rotation data augmentation. Use the CIFAR-10-Warehouse testbed to generate rotated images for testing. Train the model on the original CIFAR-10 dataset and then test it on both the original and rotated images. Measure accuracy as the metric. Compare the accuracy drop when the model is tested on rotated images with and without rotation data augmentation during training. Use a test set of at least 200 examples for each condition.

## Código de preparação de dados
```python
import pandas as pd
from datasets import load_dataset

# Load the CIFAR-10 dataset from the verified source
dataset = load_dataset('uoft-cs/cifar10', 'plain_text')

# Check the columns and print a sample
print("Columns:", dataset['train'].column_names)
print("Sample data:", dataset['train'][0])

# Split the dataset into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Extract the first 200 examples from the test set
test_subset = test_dataset.select(range(200))

# Convert the dataset to a pandas DataFrame
train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_subset)

# Display the first few rows of the training and test DataFrames
print("Training DataFrame columns:", train_df.columns)
print("Training DataFrame sample:", train_df.head())
print("Test DataFrame columns:", test_df.columns)
print("Test DataFrame sample:", test_df.head())
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
