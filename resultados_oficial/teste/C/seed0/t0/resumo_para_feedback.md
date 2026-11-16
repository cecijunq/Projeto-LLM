# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: FALHOU
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- literature_review: fase_falhou
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
1. Load the CIFAR-10 dataset using torchvision and split it into training and test sets.
2. Generate noisy versions of the test images by adding random pixel noise at levels of 10%, 20%, 30%, 40%, and 50%.
3. Train a CNN (e.g., ResNet-18) and a transformer-based model (e.g., ViT) on the clean training data.
4. Evaluate both models on the clean test set and on each noisy test set, recording their accuracy.
5. Compare the accuracy of the CNN and transformer model across the different noise levels to determine which is more sensitive to pixel noise.
6. Ensure reproducibility by fixing the random seed and performing statistical tests (e.g., paired t-test) to validate the significance of results.
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
