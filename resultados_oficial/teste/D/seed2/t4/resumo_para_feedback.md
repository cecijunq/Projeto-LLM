# Projeto: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- running_experiments: timeout
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: timeout
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2507.20019v1] This paper presents a meta-learning framework for anomaly detection in human language, treating it as a few-shot binary classification problem. The framework leverages episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that the method outperforms strong baselines in F1 and AUC scores, making it highly relevant to the study of how example order affects the performance of language models in text classification tasks.
[2205.15014v1] This paper proposes a Task-Prior Conditional Variational Auto-Encoder model named TP-VAE for few-shot image classification, particularly addressing nonuniform few-shot learning scenarios where the number of samples per class varies. The model is conditioned on support shots and constrained by a task-level prior regularization, achieving high performance in challenging nonuniform few-shot settings. It also outperforms state-of-the-art methods in standard few-shot image classification scenarios, with a notable increase in 1-shot accuracy by about 3\%. While focused on image classification, this work is relevant to understanding how example order and distribution affect model performance in few-shot learning, which is pertinent to the study of language models in text classification.
[2008.06239v2] This paper explores the application of language models as few-shot learners in task-oriented dialogue systems, focusing on how these models can be effectively fine-tuned with minimal examples. The study highlights the importance of example order in influencing the model's performance during few-shot learning. It demonstrates that language models, such as GPT-2 and GPT-3, can perform few-shot learning by simply priming the model with a few examples, which is highly relevant to understanding how example order impacts the accuracy of language models in text classification tasks. The findings contribute to the broader discussion on the role of example order in few-shot learning settings, particularly in the context of language models.
[2311.14544v1] This paper introduces a novel approach to enhance the robustness of few-shot learning models by leveraging text-derived statistics to infer latent class features. The method is designed to address the limitations of existing foundation models like CLIP, which struggle with cross-domain robustness in few-shot settings. By treating text as a source of information to predict the mean and covariance of visual features for each class, this approach enriches the latent space, leading to more generalizable and robust few-shot learning models. The paper demonstrates that incorporating both mean and covariance statistics significantly improves performance in few-shot scenarios, making it highly relevant to the study of how example order and distribution affect model performance in few-shot learning, which is pertinent to the study of language models in text classification tasks.
[2507.20019v1] This paper presents a meta-learning framework for anomaly detection in human language, treating it as a few-shot binary classification problem. The framework leverages episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that the method outperforms strong baselines in F1 and AUC scores, making it highly relevant to the study of how example order affects the performance of language models in text classification tasks.

## Plano
{
  "objective": "Investigate whether the order of few-shot examples affects the accuracy of a language model on text classification.",
  "dataset": {
    "name": "IMDB",
    "source": "HuggingFace",
    "split": {
      "train": 20000,
      "test": 200
    },
    "task": "text classification (positive/negative)"
  },
  "model": {
    "name": "llama2",
    "source": "Ollama",
    "setup": "No fine-tuning, use as-is"
  },
  "experimental_design": {
    "conditions": [
      {
        "name": "no_few_shot",
        "description": "Prompt without any few-shot examples"
      },
      {
        "name": "random_order",
        "description": "Prompt with 5 few-shot examples in random order"
      },
      {
        "name": "ordered_by_label",
        "description": "Prompt with 5 few-shot examples ordered by class label (positive first, then negative)"
      }
    ],
    "procedure": "For each test example, create three prompts with the above conditions. Generate the model's response, extract the predicted class, and record accuracy."
  },
  "evaluation": {
    "metric": "Accuracy",
    "test_set_size": 200,
    "comparison": "Compare accuracy across conditions using a paired t-test (or Wilcoxon signed-rank test if non-normal)."
  },
  "analysis": {
    "interpretation": "Interpret results in the context of previous literature on few-shot learning and example order.",
    "notes": "Avoid trivial questions (e.g., 'What is 2 + 2?') to ensure meaningful results."
  }
}

## Código de preparação de dados
```python
import os
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Print columns and an example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Extract text and label columns
texts = [example['text'] for example in dataset]
labels = [example['label'] for example in dataset]

# Print the first few texts and labels
print("\nFirst few texts and labels:")
for i in range(5):
    print(f"Text {i+1}: {texts[i]}")
    print(f"Label {i+1}: {labels[i]}\n")
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
