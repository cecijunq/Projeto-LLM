# Projeto: Are sentiment classifiers sensitive to negation words on short movie reviews (SST-2)?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- report_writing: sem_relatorio

## Artigos da revisão (3)
[1906.07610v2] This paper, "Improving Sentiment Analysis with Multi-task Learning of Negation," investigates how sentiment classifiers can be improved by explicitly incorporating information about negation. The authors propose a multi-task approach where negation is treated as an auxiliary task, and they demonstrate that this method outperforms implicit learning of negation in sentiment analysis. The study is relevant to the topic of whether sentiment classifiers are sensitive to negation words, as it directly addresses the impact of negation on sentiment prediction. The experiments are conducted on standard sentiment analysis datasets, which makes the findings applicable to tasks like SST-2.
[1712.01794v1] This paper, "The Effect of Negators, Modals, and Degree Adverbs on Sentiment Composition," investigates how negators, modals, and degree adverbs influence the sentiment of the words they modify. The study highlights that the impact of these modifiers varies significantly, and simple heuristics are insufficient to capture their true effect on sentiment. The authors created a dataset of phrases with various modifiers and their combinations, annotated with real-valued sentiment scores. Their analysis shows that the effect of modifiers varies among members of the same group, and they emphasize the importance of modeling these modifiers explicitly for accurate sentiment composition. This paper is directly relevant to the topic of whether sentiment classifiers are sensitive to negation words, as it provides insights into the complex interactions between modifiers and sentiment.
[1712.01794v1] This paper, "The Effect of Negators, Modals, and Degree Adverbs on Sentiment Composition," investigates how negators, modals, and degree adverbs influence the sentiment of the words they modify. The study highlights that the impact of these modifiers varies significantly, and simple heuristics are insufficient to capture their true effect on sentiment. The authors created a dataset of phrases with various modifiers and their combinations, annotated with real-valued sentiment scores. Their analysis shows that the effect of modifiers varies among members of the same group, and they emphasize the importance of modeling these modifiers explicitly for accurate sentiment composition. This paper is directly relevant to the topic of whether sentiment classifiers are sensitive to negation words, as it provides insights into the complex interactions between modifiers and sentiment.

## Plano
{
  "objective": "Investigate whether sentiment classifiers are sensitive to negation words on short movie reviews (SST-2).",
  "models": {
    "model_name": "llama3",
    "provider": "ollama"
  },
  "data": {
    "dataset": "sst-2",
    "source": "huggingface"
  },
  "steps": [
    {
      "step": "1",
      "action": "Load the SST-2 dataset from HuggingFace and split it into training and test sets.",
      "purpose": "Prepare the dataset for fine-tuning and evaluation."
    },
    {
      "step": "2",
      "action": "Fine-tune the Llama3 model on the training set.",
      "purpose": "Train the model to classify sentiment on the SST-2 dataset."
    },
    {
      "step": "3",
      "action": "Evaluate the model on the test set to get baseline performance.",
      "purpose": "Establish a baseline for sentiment classification without negation words."
    },
    {
      "step": "4",
      "action": "Create a modified test set by adding negation words (e.g., 'not', 'never', 'hardly') to sentences.",
      "purpose": "Introduce negation words to test the model's sensitivity to such modifiers."
    },
    {
      "step": "5",
      "action": "Re-evaluate the model on the modified test set and compare performance with the baseline.",
      "purpose": "Determine if the model is sensitive to negation words."
    },
    {
      "step": "6",
      "action": "Analyze the results to draw conclusions about the model's sensitivity to negation words.",
      "purpose": "Provide insights into whether sentiment classifiers are sensitive to negation in short movie reviews."
    }
  ],
  "constraints": {
    "time_limit": "few minutes",
    "resources": "one GPU",
    "data_size": "small",
    "model": "local language model served by Ollama",
    "no_external_apis": true
  }
}

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
