# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (1)
[2112.09300v1] We propose an end-to-end image compression and analysis model with Transformers, targeting cloud-based image classification applications. The model redesigns the Vision Transformer (ViT) to perform classification from compressed features, leveraging long-term information from the Transformer. A lightweight convolutional neural network is used to replace the patchify stem of ViT, injecting convolutional inductive bias into the Transformer for improved performance. This work highlights the integration of compression and analysis, which is relevant to understanding how Transformers handle noise in compressed data, indirectly relating to pixel noise resilience in image classification tasks.

## Plano
1. Load and preprocess the MNIST dataset using torchvision, splitting it into training and test sets. 
2. Introduce pixel noise to the images by randomly flipping 10%, 20%, and 30% of the pixel values to 0 or 255 (for grayscale). 
3. Train a simple CNN (e.g., 2 convolutional layers followed by dense layers) using PyTorch on the clean MNIST dataset. 
4. Load a pre-trained ViT model using Ollama's local serving API (e.g., using the "llava" or "vicuna" model if available for image tasks). 
5. Evaluate both models on the noisy test images, calculating accuracy and F1 scores for each noise level. 
6. Compare the performance degradation of the CNN and ViT to determine which model is more sensitive to pixel noise. 
7. Document the results and analyze whether the ViT's transformer architecture provides better noise resilience compared to the CNN.
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
