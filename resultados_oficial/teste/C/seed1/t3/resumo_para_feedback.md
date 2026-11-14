# Projeto: Does rotation data augmentation make a small CNN more robust to rotated images on CIFAR-10?

## Etapas
- literature_review: FALHOU
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- literature_review: fase_falhou
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
1. **Objective**: Investigate whether rotation data augmentation improves the robustness of a small CNN on rotated images in the CIFAR-10 dataset.

2. **Models**: 
   - Use a small CNN architecture (e.g., 3 convolutional layers with ReLU and max-pooling, followed by a fully connected layer).
   - Two variants: one with rotation data augmentation (e.g., rotating images by 90°, 180°, 270° during training) and one without.

3. **Data Source**: 
   - Use the CIFAR-10 dataset from torchvision.

4. **Experimental Steps**: 
   - Load the CIFAR-10 dataset and split it into training and test sets (e.g., 80% training, 20% test).
   - For the augmented model, apply rotation transformations (e.g., 90°, 180°, 270°) to the training images using torchvision.transforms.
   - Train both models on the training set with the same hyperparameters (e.g., optimizer, learning rate, number of epochs).
   - Evaluate both models on the test set using standard metrics (e.g., accuracy, F1-score).
   - Test robustness by applying rotations (e.g., 90°, 180°, 270°) to the test images and evaluate the performance of both models.
   - Compare the results to determine if rotation augmentation improves robustness.
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
