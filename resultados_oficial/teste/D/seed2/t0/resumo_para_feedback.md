# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2112.09300v1] This paper proposes an end-to-end image compression and analysis model using Transformers, aiming to improve image classification from compressed features. While the paper does not directly compare noise sensitivity between Transformers and CNNs, it highlights the potential of Transformers to leverage long-term information, which may offer advantages in handling noisy data. The study suggests that Transformers could be more robust in scenarios where global context is important, which is relevant to understanding noise sensitivity differences.
[2210.15722v1] This paper introduces PatchRot, a self-supervised technique designed to train Vision Transformers (ViTs) effectively without the need for labeled data. PatchRot involves rotating images and patches and training the network to predict rotation angles, which enables the model to learn both global and local features. While the paper does not directly address noise sensitivity, the method's emphasis on learning robust features from corrupted or rotated data implies that ViTs trained with such techniques may exhibit improved resilience to pixel noise compared to traditional convolutional networks. This aligns with the broader discussion on how self-supervised learning can enhance the noise tolerance of vision models.
[2507.21156v1] This paper presents a comprehensive comparative analysis of Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) for medical image classification tasks. The study evaluates their performance on three critical medical imaging tasks: chest X-ray pneumonia detection, brain tumor classification, and skin cancer melanoma detection. The results show that CNNs like ResNet-50 achieve high accuracy on chest X-ray classification (98.37%), while ViTs like DeiT-Small excel in brain tumor detection (92.16%). The paper highlights task-specific advantages of each architecture, suggesting that CNNs may be more robust to noise in certain medical imaging scenarios, while ViTs can offer better performance in tasks requiring global context. This aligns with the broader discussion on the noise sensitivity of different vision models.
[2010.01177v4] This paper introduces the Global Adaptive Filtering Layer, a novel neural layer designed to enhance the performance of computer vision models by adaptively learning optimal frequency filters for each image. The layer operates in the frequency domain, selecting the best frequencies for each image and feeding the processed data back to the main neural network. This approach significantly improves the robustness of models against various types of noise, including pixel noise. The study demonstrates that incorporating this adaptive layer can enhance the noise tolerance of both convolutional networks and vision transformers, suggesting that models equipped with such layers may be more resilient to pixel noise compared to their baseline versions. This finding is relevant to understanding the noise sensitivity of different vision models in image classification tasks.
[2103.13023v1] This paper investigates whether Vision Transformers (ViTs) can be effectively pre-trained without natural images and human-annotated labels. While the paper does not directly address noise sensitivity, it explores the feasibility of training ViTs in an unsupervised manner, which could indirectly relate to their robustness to noise. The study demonstrates that a ViT pre-trained without natural images can partially outperform existing self-supervised learning methods like SimCLRv2 and MoCov2. This finding suggests that ViTs may have the potential to be more resilient to various forms of data corruption, including noise, if trained in a manner that emphasizes robust feature learning. This aligns with the broader discussion on the noise sensitivity of vision models.

## Plano
Evaluate the noise sensitivity of Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. Add pixel noise at levels of 0%, 5%, 10%, 15%, 20%, 25%, and 30% to the test images. For each noise level, compute the accuracy of both ViT-base and ResNet-50 on the noisy images. Use the same test set for both models and report the accuracy drop from the clean baseline. Compare the noise sensitivity by analyzing the accuracy drop across the noise levels for both models.

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
