# Projeto: Does chain-of-thought prompting improve the accuracy of a small language model on grade-school math (GSM8K)?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- running_experiments: timeout
- running_experiments: timeout
- running_experiments: timeout
- running_experiments: timeout
- running_experiments: timeout
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2201.11903v6] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
[2409.17904v1] Learning to Love Edge Cases in Formative Math Assessment: Using the AMMORE Dataset and Chain-of-Thought Prompting to Improve Grading Accuracy
[2311.09277v1] Contrastive Chain-of-Thought Prompting
[2309.11054v2] Design of Chain-of-Thought in Math Problem Solving
[2609.39049v2] Structure vs. Chain-of-Thought: Evaluating LLM Criteria Extraction for Depression Severity

## Plano
{
  "objective": "Investigate whether chain-of-thought (CoT) prompting improves the accuracy of a small language model on grade-school math problems from the GSM8K dataset.",
  "methodology": {
    "dataset": "gsm8k (from HuggingFace)",
    "model": "Ollama-local model (via OpenAI-compatible API)",
    "steps": [
      "Load the GSM8K dataset and split into training (80%) and test (20%) sets.",
      "Evaluate the model's performance on the test set using standard prompting.",
      "Repeat the evaluation using CoT prompting, where each problem is followed by a step-by-step reasoning process.",
      "Compare the accuracy of the model under both prompting conditions."
    ]
  },
  "resources": {
    "data": "Small dataset (GSM8K) from HuggingFace",
    "model": "Ollama-local model (no external APIs or model downloads)",
    "time": "A few minutes on one GPU"
  },
  "analysis": "Compare the model's accuracy with and without CoT prompting to determine if CoT improves performance on GSM8K."
}

## Código de preparação de dados
```python
import pandas as pd
from datasets import load_dataset

# Load the GSM8K dataset with the 'main' configuration and split into test set
dataset = load_dataset('openai/gsm8k', 'main', split='test[:200]')

# Convert to a pandas DataFrame for easier handling
df = dataset.to_pandas()

# Display the first few rows to confirm the data structure
print(df.head())

# Save the dataset to a CSV file for further processing
df.to_csv('gsm8k_test_data.csv', index=False)
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
