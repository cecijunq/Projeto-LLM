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
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- report_writing: sem_relatorio

## Artigos da revisão (1)
[1906.07610v2] This paper proposes a multi-task learning approach to improve sentiment analysis by explicitly incorporating information about negation. The study focuses on short texts, such as movie reviews, and demonstrates that explicitly modeling negation outperforms implicit learning of negation in a data-driven manner. The results show that sentiment classifiers are sensitive to negation words, and the proposed approach significantly improves the accuracy of sentiment prediction in the presence of negation.

## Plano
1. Load the SST-2 dataset from HuggingFace.
2. Split the dataset into training and test sets.
3. Use a pre-trained language model (e.g., llama3) via Ollama to perform inference on the test set.
4. Evaluate the model's performance by predicting sentiment for each review.
5. Repeat the experiment by adding a negation marker (e.g., "NOT") to the beginning of each review containing negation words.
6. Compare the accuracy and F1-score of the two experiments to assess the model's sensitivity to negation words.
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
