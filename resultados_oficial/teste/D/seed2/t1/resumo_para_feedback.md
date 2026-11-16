# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2405.11357v3] Large language models (LLMs) have demonstrated remarkable performances on a wide range of natural language tasks. Yet, LLMs' successes have been largely restricted to tasks concerning words, sentences, or documents, and it remains questionable how much they understand the minimal units of text, namely characters. This paper examines contemporary LLMs regarding their ability to understand character composition of words, showing that most of them fail to reliably carry out even the simple tasks that can be handled by humans with perfection. The study analyzes their behaviors with comparison to token level performances and discusses potential directions for future research.
[2405.11357v3] Large language models (LLMs) have demonstrated remarkable performances on a wide range of natural language tasks. Yet, LLMs' successes have been largely restricted to tasks concerning words, sentences, or documents, and it remains questionable how much they understand the minimal units of text, namely characters. This paper examines contemporary LLMs regarding their ability to understand character composition of words, showing that most of them fail to reliably carry out even the simple tasks that can be handled by humans with perfection. The study analyzes their behaviors with comparison to token level performances and discusses potential directions for future research.
[2405.11357v3] Large language models (LLMs) have demonstrated remarkable performances on a wide range of natural language tasks. Yet, LLMs' successes have been largely restricted to tasks concerning words, sentences, or documents, and it remains questionable how much they understand the minimal units of text, namely characters. This paper examines contemporary LLMs regarding their ability to understand character composition of words, showing that most of them fail to reliably carry out even the simple tasks that can be handled by humans with perfection. The study analyzes their behaviors with comparison to token level performances and discusses potential directions for future research.
[2405.11357v3] Large language models (LLMs) have demonstrated remarkable performances on a wide range of natural language tasks. Yet, LLMs' successes have been largely restricted to tasks concerning words, sentences, or documents, and it remains questionable how much they understand the minimal units of text, namely characters. This paper examines contemporary LLMs regarding their ability to understand character composition of words, showing that most of them fail to reliably carry out even the simple tasks that can be handled by humans with perfection. The study analyzes their behaviors with comparison to token level performances and discusses potential directions for future research.
[2405.11357v3] Large language models (LLMs) have demonstrated remarkable performances on a wide range of natural language tasks. Yet, LLMs' successes have been largely restricted to tasks concerning words, sentences, or documents, and it remains questionable how much they understand the minimal units of text, namely characters. This paper examines contemporary LLMs regarding their ability to understand character composition of words, showing that most of them fail to reliably carry out even the simple tasks that can be handled by humans with perfection. The study analyzes their behaviors with comparison to token level performances and discusses potential directions for future research.

## Plano
We will design an experiment to test whether language models are sensitive to word order in multiple choice benchmarks. We will use the AGQA dataset from HuggingFace, which contains questions and answers. For each question, we will permute the word order of the question and the choices while keeping the meaning intact. We will evaluate the model's accuracy on both the original and permuted versions of the questions. The same model served by Ollama will be used to ensure consistency. We will use a test set of at least 200 examples and measure the accuracy of the model on both versions of the questions. We will compare the accuracy of the model on the original and permuted versions of the questions to determine if the model is sensitive to word order.

## Código de preparação de dados
```python
import pandas as pd
from datasets import load_dataset

# Load dataset
dataset = load_dataset('mateus-hamade/multiple-choice-questions', split='train[:200]')

# Display column names and a sample entry
print("Columns:", dataset.column_names)
print("Sample entry:")
print(dataset[0])

# Convert to pandas DataFrame for easier handling
df = pd.DataFrame(dataset)

# Display first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())
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
