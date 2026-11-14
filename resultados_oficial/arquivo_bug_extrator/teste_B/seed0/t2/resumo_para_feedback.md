# Projeto: Are sentiment classifiers sensitive to negation words on short movie reviews (SST-2)?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (2)
[1906.07610v2] This paper investigates the impact of negation words on sentiment analysis, particularly in short texts. The authors propose a multi-task learning approach that explicitly incorporates negation as an auxiliary task, which helps improve the main task of sentiment analysis. They demonstrate that their cascading neural architecture with selective sharing of LSTM layers outperforms implicit learning of negation in a data-driven manner. The study is evaluated on the SST-2 dataset, which consists of short movie reviews, and shows that explicitly training the model with negation as an auxiliary task improves the model's sensitivity to negation words, leading to better performance in sentiment classification.
[1406.2022v1] This paper introduces a novel approach to sentiment analysis by classifying text into a two-dimensional emotional space based on the sentiments of the author. The method utilizes existing lexical resources to extract features, which are then trained using supervised learning techniques. While the paper does not focus specifically on negation handling or short texts like the SST-2 dataset, it provides a foundational perspective on sentiment analysis that can be extended to incorporate negation as a key aspect of sentiment interpretation.

## Plano
Step 1: Load the SST-2 dataset from HuggingFace using the Ollama model for text processing.  
Step 2: Preprocess the dataset by tokenizing the text and converting labels to numerical values.  
Step 3: Train a baseline sentiment classifier (e.g., logistic regression) on the dataset.  
Step 4: Train a negation-aware model using a multi-task learning approach, where the model learns to detect negation words as an auxiliary task.  
Step 5: Evaluate both models on the test set using accuracy and F1 score.  
Step 6: Analyze the results to determine whether the negation-aware model shows improved sensitivity to negation words on short movie reviews.  
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
