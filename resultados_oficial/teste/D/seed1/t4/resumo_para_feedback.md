# Projeto: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2205.15014v1] This paper introduces the Task-Prior Conditional Variational Auto-Encoder (TP-VAE) model for few-shot image classification. TP-VAE is designed to handle nonuniform few-shot learning scenarios, where the number of samples per class is not uniform. The model is conditioned on support shots and constrained by a task-level prior regularization, which allows it to perform well in both nonuniform and standard few-shot settings. The method outperforms state-of-the-art approaches, achieving significant improvements in 1-shot accuracy. While the paper focuses on image classification, the concepts of task-level priors and conditional variational auto-encoders could be relevant to understanding how the order of examples might influence model performance in text classification tasks.
[2507.20019v1] This paper proposes a meta learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language ranging from spam and fake news to hate speech pose a major challenge due to their sparsity and variability. The paper treats anomaly detection as a few shot binary classification problem and leverages meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, the method evaluates model generalization on unseen tasks with minimal labeled anomalies. The framework combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results demonstrate that the method outperforms strong baselines in F1 and AUC scores. The approach could be relevant to understanding how the order of examples might influence model performance in text classification tasks, particularly in detecting rare or novel patterns.
[2311.14544v1] This paper introduces a novel approach to robust visual few-shot learning by leveraging text-derived statistics to predict the mean and covariance of the visual feature distribution for each class. The method enhances foundation models like CLIP by incorporating text as a modality to infer latent class statistics, which helps in understanding the underlying visual feature distribution. This predictive framework enriches the latent space, leading to more robust and generalizable few-shot learning models. The paper demonstrates that incorporating both mean and covariance statistics significantly improves few-shot learning performance, particularly in cross-domain scenarios. While the paper focuses on visual few-shot learning, the concept of using text to infer latent statistics could be relevant to understanding how the order of examples might influence model performance in text classification tasks.
[2008.06239v2] This paper explores the application of language models as few-shot learners for task-oriented dialogue systems. The study emphasizes the challenge of learning each module (Natural Language Understanding, Dialogue State Tracking, Dialogue Policy, and Natural Language Generation) with minimal data samples. The paper highlights how large pre-trained language models, such as GPT-2 and GPT-3, can perform few-shot learning by providing a few examples as prompts. This approach eliminates the need for extensive fine-tuning and parameter adjustment for each task. The paper demonstrates the effectiveness of few-shot learning in dialogue systems, suggesting that the order of examples might influence model performance, particularly in scenarios where the sequence of interactions is critical for accurate classification and response generation. This work is relevant to understanding how the order of examples might influence model performance in text classification tasks.
[2507.20019v1] This paper proposes a meta learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language ranging from spam and fake news to hate speech pose a major challenge due to their sparsity and variability. The paper treats anomaly detection as a few shot binary classification problem and leverages meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, the method evaluates model generalization on unseen tasks with minimal labeled anomalies. The framework combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results demonstrate that the method outperforms strong baselines in F1 and AUC scores. The approach could be relevant to understanding how the order of examples might influence model performance in text classification tasks, particularly in detecting rare or novel patterns.

## Plano
We will test the hypothesis that the order of few-shot examples changes the accuracy of a language model on text classification. The experiment will use the SMS Spam Collection dataset (200 examples) and Ollama's Llama 3 model. We will evaluate three conditions: (1) original order, (2) reversed order, and (3) shuffled order. For each condition, we will construct a few-shot prompt with 10 examples and evaluate the model on the test set. Accuracy will be measured for each condition. We will analyze whether the order of examples affects classification performance.

## Código de preparação de dados
```python
import os
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset('argilla/end2end_textclassification', split='train[:200]')

# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Print columns and an example
print("Columns:", df.columns)
print("Example:", df.iloc[0].to_dict())

# Extract text and label columns
text_column = 'text'
label_column = 'label'

# Check if the required columns exist
if text_column not in df.columns or label_column not in df.columns:
    raise ValueError(f"Dataset missing required columns: {text_column} or {label_column}")

# Prepare data for the experiment
print("Data preparation complete. Ready to proceed with the experiment.")
```

## Código do experimento
```python

```

## Saída do experimento
```
Nenhum código executou com sucesso.
```

## Interpretação
(sem interpretação)

## Relatório final

(sem relatório)
