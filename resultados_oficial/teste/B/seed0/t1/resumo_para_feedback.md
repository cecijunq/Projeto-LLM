# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (1)
[2308.11483v1] Large Language Models (LLMs) have demonstrated remarkable capabilities in various NLP tasks. However, previous works have shown these models are sensitive towards prompt wording, and few-shot demonstrations and their order, posing challenges to fair assessment of these models. As these models become more powerful, it becomes imperative to understand and address these limitations. In this paper, we focus on LLMs robustness on the task of multiple-choice questions -- commonly adopted task to study reasoning and fact-retrieving capability of LLMs. Investigating the sensitivity of LLMs towards the order of options in multiple-choice questions, we demonstrate a considerable performance gap of approximately 13% to 75% in LLMs on different benchmarks, when answer options are reordered, even when the content remains the same. This sensitivity highlights the importance of carefully designing and evaluating multiple-choice benchmarks to ensure fair and reliable assessment of LLMs.

## Plano
1. **Dataset Preparation**: Use a subset of the MMLU dataset (e.g., 100 questions from a single subject) to ensure the experiment runs quickly on a single GPU. Include the question and four answer choices (A, B, C, D).

2. **Randomization**: For each question, shuffle the order of the answer choices while keeping the content (text) of each option the same. Save the shuffled versions as a separate dataset.

3. **Model Evaluation**: Use the Ollama API to query the local language model with both the original and shuffled versions of the questions. For each question, record the model's selected answer.

4. **Performance Comparison**: Calculate the accuracy of the model on both the original and shuffled datasets. Compare the two accuracy scores to determine if the order of the answer choices significantly impacts performance.

5. **Analysis**: Interpret the results to assess whether the language model is sensitive to the order of answer choices in multiple-choice tasks. If a significant performance difference is observed, this would support the hypothesis that models are sensitive to word order in such benchmarks.
</final plan>

## Código de preparação de dados
```python
from datasets import load_dataset
import random

# Load the MMLU dataset with a subset of 200 questions from the 'test' split
dataset = load_dataset('cais/mmlu', 'all', split='test[:200]')

# Check if the required columns are present
required_columns = ['question', 'subject', 'choices', 'answer']
missing_columns = [col for col in required_columns if col not in dataset.column_names]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

# Shuffle the order of answer choices for each question
shuffled_dataset = dataset.map(lambda example: {
    'question': example['question'],
    'subject': example['subject'],
    'choices': random.sample(example['choices'], len(example['choices'])),
    'answer': example['answer']
})

# Save the shuffled dataset to a temporary file
shuffled_dataset.save_to_disk("shuffled_dataset")
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
