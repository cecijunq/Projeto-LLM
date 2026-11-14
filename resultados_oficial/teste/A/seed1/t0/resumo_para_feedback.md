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
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- data_preparation: timeout
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
1. Load CIFAR-10 dataset and split into train/val/test sets.
2. Preprocess images by normalizing to [0,1].
3. Add Gaussian noise with standard deviations ranging from 0.05 to 0.25 (5% to 25% of pixel range).
4. Use a pre-trained ResNet-18 (convolutional network) and ViT-base (image transformer) from HuggingFace.
5. Evaluate accuracy of both models on noisy test images.
6. Plot accuracy vs. noise level to compare sensitivity to pixel noise.
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
