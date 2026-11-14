# Projeto: Are image transformers more or less sensitive to pixel noise than convolutional networks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: formato_invalido
- literature_review: busca_em_loop
- report_writing: sem_relatorio

## Artigos da revisão (2)
[1806.06357v1] StegNet explores the robustness of deep convolutional networks in image steganography, where it embeds hidden information with minimal distortion. The paper shows that the method is robust to statistical analysis, implying that convolutional networks can maintain image quality even with added noise or modifications. This provides insight into the noise sensitivity of convolutional networks in practical scenarios.
[2111.10480v6] TransMorph introduces a hybrid Transformer-ConvNet model for volumetric medical image registration, highlighting the potential of Transformers to address the limitations of convolutional networks in capturing long-range spatial relationships. The paper demonstrates that Transformers can achieve state-of-the-art performance in medical imaging applications by leveraging their large receptive fields for precise spatial correspondence. This provides insights into how Transformers may be more robust to noise compared to traditional convolutional networks in tasks requiring spatial understanding.

## Plano
1. Load the FashionMNIST dataset using torchvision, split into training and test sets.
2. Define a simple ConvNet and a Vision Transformer (ViT) model for image classification.
3. Introduce varying levels of Gaussian noise (5%, 10%, 15%, 20%) to the test images.
4. Evaluate the models' accuracy on the noisy images using Ollama's local language model.
5. Compare the accuracy of both models across noise levels to determine which is more robust to pixel noise.
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
