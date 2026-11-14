# Projeto: Does rotation data augmentation make a small CNN more robust to rotated images on CIFAR-10?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: formato_invalido
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 datasets with diverse environmental discrepancies. These datasets are collected using image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This work is relevant to the topic as it explores model robustness across different environments, including potential applications of data augmentation techniques like rotation on CIFAR-10.
[2411.15497v3] AeroGen enhances remote sensing object detection by introducing a layout-controllable diffusion generative model tailored for remote sensing image object detection (RSIOD). This model supports the generation of both horizontal and rotated bounding boxes, enabling the creation of high-quality synthetic data. While not directly focused on CNN robustness or rotation data augmentation on CIFAR-10, AeroGen's approach to generating rotated objects through diffusion models is conceptually relevant to understanding how rotation augmentation can improve model robustness in tasks involving rotated inputs.
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 datasets with diverse environmental discrepancies. These datasets are collected using image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This work is relevant to the topic as it explores model robustness across different environments, including potential applications of data augmentation techniques like rotation on CIFAR-10.
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 datasets with diverse environmental discrepancies. These datasets are collected using image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This work is relevant to the topic as it explores model robustness across different environments, including potential applications of data augmentation techniques like rotation on CIFAR-10.
[2310.04414v3] CIFAR-10-Warehouse provides a broad and realistic testbed for analyzing model generalization by introducing 180 datasets with diverse environmental discrepancies. These datasets are collected using image search engines and diffusion models, covering natural images, cartoons, specific colors, and objects that do not naturally appear. This work is relevant to the topic as it explores model robustness across different environments, including potential applications of data augmentation techniques like rotation on CIFAR-10.

## Plano
1. **Models**: 
   - A small CNN (e.g., a 3-layer CNN with 32 filters in the first layer, 64 in the second, and 128 in the third, followed by global average pooling and a softmax classifier).
   - A baseline model trained without rotation augmentation.

2. **Data Source**: 
   - CIFAR-10 dataset from HuggingFace (dataset id: 'torchvision/cifar10').

3. **Experimental Steps**:
   - Split the CIFAR-10 dataset into training and test sets (e.g., 50,000 training images and 10,000 test images).
   - Generate rotated versions of the training images (e.g., 0°, 90°, 180°, 270°) for data augmentation.
   - Train two models:
     - Model A: Trained with rotation data augmentation (images rotated by 0°, 90°, 180°, 270° during training).
     - Model B: Trained without any rotation augmentation.
   - Test both models on a test set of rotated images (e.g., images rotated by 15°, 30°, 45°, 60°, 75°, and 90°).
   - Evaluate the accuracy of both models on the rotated test images.
   - Compare the performance of the two models to determine if rotation data augmentation improves robustness to rotated images.
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
