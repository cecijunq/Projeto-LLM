# Projeto: Does rotation data augmentation make a small CNN more robust to rotated images on CIFAR-10?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: formato_invalido
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 diverse datasets generated through image search engines and diffusion models. These datasets cover a wide range of environmental discrepancies, including natural images, cartoons, specific colors, and objects that do not naturally appear. This paper is relevant to the topic as it explores the performance of models on out-of-distribution data, which is crucial for understanding the robustness of CNNs to data augmentation techniques like rotation on CIFAR-10.
[2411.15497v3] AeroGen is a layout-controllable diffusion generative model specifically designed for remote sensing image object detection (RSIOD). It addresses the challenge of limited labeled data in RSIOD by enabling the generation of high-quality synthetic images with horizontal and rotated bounding box conditions. This paper is relevant to the topic as it explores generative data augmentation techniques for improving model performance on specific image transformations, which is analogous to the study of rotation data augmentation in CIFAR-10.
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 diverse datasets generated through image search engines and diffusion models. These datasets cover a wide range of environmental discrepancies, including natural images, cartoons, specific colors, and objects that do not naturally appear. This paper is relevant to the topic as it explores the performance of models on out-of-distribution data, which is crucial for understanding the robustness of CNNs to data augmentation techniques like rotation on CIFAR-10.
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 diverse datasets generated through image search engines and diffusion models. These datasets cover a wide range of environmental discrepancies, including natural images, cartoons, specific colors, and objects that do not naturally appear. This paper is relevant to the topic as it explores the performance of models on out-of-distribution data, which is crucial for understanding the robustness of CNNs to data augmentation techniques like rotation on CIFAR-10.
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 diverse datasets generated through image search engines and diffusion models. These datasets cover a wide range of environmental discrepancies, including natural images, cartoons, specific colors, and objects that do not naturally appear. This paper is relevant to the topic as it explores the performance of models on out-of-distribution data, which is crucial for understanding the robustness of CNNs to data augmentation techniques like rotation on CIFAR-10.

## Plano
1. Load the CIFAR-10 dataset using torchvision, ensuring the training and test sets are split.
2. Create two data loaders: one with random rotation augmentation (e.g., 15 degrees) and one without.
3. Define a simple CNN architecture (e.g., 5 layers) and train two models: one with rotation augmentation and one without, using the same hyperparameters (e.g., learning rate, batch size, number of epochs).
4. Generate a test set of CIFAR-10 images rotated by 15 degrees.
5. Evaluate both models on the rotated test set and compute their accuracy.
6. Compare the accuracy of the two models to determine if rotation data augmentation improves robustness to rotated images.
</final plan>

## Código de preparação de dados
```python

```

## Código do experimento
```python

```

## Saída do experimento
```

```

## Interpretação
(sem interpretação)

## Relatório final

(sem relatório)
