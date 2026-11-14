# Projeto: Are sentiment classifiers sensitive to negation words on short movie reviews (SST-2)?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: erro_de_sintaxe
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- report_writing: sem_relatorio

## Artigos da revisão (3)
[1906.07610v2] This paper, "Improving Sentiment Analysis with Multi-task Learning of Negation," investigates the impact of negation on sentiment classification. The authors propose a multi-task learning approach where negation is treated as an auxiliary task, alongside the main sentiment classification task. They demonstrate that explicitly modeling negation improves the performance of sentiment analysis models. The paper uses a cascading neural architecture with selective sharing of LSTM layers and reports improved results on sentiment classification tasks when negation is explicitly incorporated.
[1906.07599v1] This paper, "LTG-Oslo Hierarchical Multi-task Network: The importance of negation for document-level sentiment in Spanish," explores the role of negation in document-level sentiment analysis. The authors propose a hierarchical multi-task network that uses shared lower layers in a deep BiLSTM to predict negation, while higher layers are dedicated to predicting document-level sentiment. The paper highlights the potential of multi-task learning to incorporate negation information into deep neural sentiment classifiers, even though the absolute results on the test set were relatively low for a binary classification task.
[2407.05627v1] This paper, "New Directions in Text Classification Research: Maximizing The Performance of Sentiment Classification from Limited Data," focuses on improving sentiment classification from limited training data. While not directly focused on negation words, the paper emphasizes the importance of handling compositional phenomena like negation for accurate sentiment analysis. The authors introduce a benchmark dataset for sentiment classification with limited data and propose methods for data augmentation. The study highlights the need for robust models that can handle the nuances of language, including negation, to achieve high accuracy in sentiment prediction, particularly on datasets like SST-2.

## Plano
1. **Objective**: Evaluate the sensitivity of the local language model (served by Ollama) to negation words in the SST-2 dataset.

2. **Data Source**: Use the SST-2 dataset from HuggingFace (`glue/sst2`). The dataset will be loaded and split into training, validation, and test sets.

3. **Models**: 
   - Use the local language model served by Ollama (e.g., `llama3`).

4. **Experimental Steps**:
   - Load the SST-2 dataset and split it into training, validation, and test sets.
   - Preprocess the test set by identifying negation words (e.g., "not," "no," "never") and marking their presence in each review. Create two subsets: one with negation words and one without.
   - Evaluate the local language model's performance on both subsets using sentiment classification (positive/negative).
   - Calculate accuracy and F1-score for both subsets to compare the model's sensitivity to negation words.
   - Analyze the results to determine whether the model is sensitive to negation in short movie reviews.

5. **Expected Outcome**: The experiment will assess how the local language model handles sentiment in the presence of negation words, providing insights into its sensitivity to such linguistic phenomena without requiring external models or datasets.
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
