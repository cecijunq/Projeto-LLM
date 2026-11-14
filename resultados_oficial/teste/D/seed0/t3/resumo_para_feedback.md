# Projeto: Does rotation data augmentation make a small CNN more robust to rotated images on CIFAR-10?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2210.08363v3] This paper introduces a data-efficient augmentation technique for training neural networks, focusing on improving generalization and reducing computational costs. The authors propose a method to select subsets of data points that, when augmented, capture the training dynamics of full data augmentation. By modeling augmentation as additive perturbations, they show that it improves learning by enlarging and perturbing smaller singular values of the network Jacobian, while preserving prominent directions. This approach prevents overfitting and enhances generalization, making it particularly relevant for resource-constrained scenarios, such as training small CNNs on datasets like CIFAR-10. The results demonstrate that data-efficient augmentation can achieve comparable performance to full augmentation while significantly reducing computational overhead.
[2108.02704v1] This paper introduces Rotaflip, a novel CNN layer designed to enhance regularization and achieve rotational invariance in medical images. The proposed layer applies random rotations and reflections to a small percentage of feature maps after each convolutional layer, which helps in capturing orientational symmetries present in medical images. This approach provides a degree of rotational invariance without the need for additional parameters or data augmentation. The method was tested on histopathology images from the PatchCamelyon dataset using a DenseNet architecture, demonstrating its effectiveness in improving rotational robustness and generalization in medical image classification tasks.
[2211.11812v1] This paper introduces RIC-CNN, a rotation-invariant convolutional neural network designed to address the limitation of standard CNNs in handling rotated images. The authors propose a new convolutional operation called Rotation-Invariant Coordinate Convolution (RIC-C), which operates in a rotation-invariant coordinate system. This allows the network to be naturally invariant to arbitrary rotations around the input center without the need for additional parameters or data augmentation. The RIC-CNN is evaluated on various datasets, including rotated versions of standard image benchmarks, and demonstrates significant improvements in performance when compared to conventional CNNs. The results highlight the effectiveness of RIC-CNN in enhancing robustness to rotation, making it a promising approach for tasks involving rotated images, such as those found in the CIFAR-10 dataset.
[2211.11812v1] This paper introduces RIC-CNN, a rotation-invariant convolutional neural network designed to address the limitation of standard CNNs in handling rotated images. The authors propose a new convolutional operation called Rotation-Invariant Coordinate Convolution (RIC-C), which operates in a rotation-invariant coordinate system. This allows the network to be naturally invariant to arbitrary rotations around the input center without the need for additional parameters or data augmentation. The RIC-CNN is evaluated on various datasets, including rotated versions of standard image benchmarks, and demonstrates significant improvements in performance when compared to conventional CNNs. The results highlight the effectiveness of RIC-CNN in enhancing robustness to rotation, making it a promising approach for tasks involving rotated images, such as those found in the CIFAR-10 dataset.
[2310.04414v3] This paper introduces CIFAR-10-Warehouse, a comprehensive testbed consisting of 180 datasets collected through various methods such as prompting image search engines and diffusion models. The datasets are designed to provide a broad and realistic representation of out-of-distribution environments, covering natural images, cartoons, specific colors, and objects that do not naturally appear. The purpose of CIFAR-10-Warehouse is to enable rigorous analysis of model generalization across diverse and realistic scenarios. This testbed is particularly relevant for evaluating the robustness of small CNNs to rotated images, as it includes datasets with variations in orientation and environmental conditions, offering a benchmark for assessing the effectiveness of rotation data augmentation techniques on the CIFAR-10 dataset.

## Plano
1. Dataset: Use the CIFAR-10 dataset from torchvision and include rotated versions of images (0°, 90°, 180°, 270°) from the CIFAR-10-Warehouse dataset (available via HuggingFace) to create a test set of at least 200 examples.
2. Model: Train two small CNNs (e.g., a standard CNN and a RIC-CNN from [2211.11812v1]) on the original CIFAR-10 dataset.
3. Training: 
   - Model A: Train with rotation data augmentation (e.g., random rotations of images during training).
   - Model B: Train without rotation data augmentation.
4. Evaluation:
   - Test both models on the original CIFAR-10 test set (no rotation).
   - Test both models on the rotated versions of the original images (0°, 90°, 18, 270°) from the CIFAR-10-Warehouse dataset.
5. Metric: Use classification accuracy as the evaluation metric.
6. Analysis: Compare the accuracy of both models on the original and rotated test sets to determine whether rotation data augmentation improves robustness to rotated images.

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
