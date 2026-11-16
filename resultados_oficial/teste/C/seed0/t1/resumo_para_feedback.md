# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: FALHOU
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- literature_review: fase_falhou
- running_experiments: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- running_experiments: erro_de_execucao
- running_experiments: timeout
- running_experiments: erro_de_execucao
- running_experiments: timeout
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
1. Load a subset of the GSM8K dataset (e.g., 100 questions) for efficiency.  
2. For each question, split the text into words, reverse the order, and reformat the question while preserving mathematical content.  
3. Use the Ollama language model to generate answers for both original and reversed questions, instructing it to end responses with 'Answer: <number>'.  
4. Extract the final number using regex and compare it numerically with the ground truth for accuracy.  
5. Compare the accuracy of the model on the original vs. reversed word order versions.  
6. Analyze results to determine if word order impacts model performance.  
</final plan>

## Código de preparação de dados
```python
from datasets import load_dataset
import re

# Load the GSM8K dataset with the specified split
dataset = load_dataset('openai/gsm8k', 'main', split='test[:200]')

# Function to reverse the word order of a question while preserving mathematical content
def reverse_question(text):
    # Split the text into parts separated by <<...>>
    parts = re.split(r'<<[^>]*?>>', text)
    reversed_parts = []
    for part in parts:
        # Split the part into words and reverse the order
        words = part.split()
        reversed_words = words[::-1]
        reversed_part = ' '.join(reversed_words)
        reversed_parts.append(reversed_part)
    # Join the parts back together
    reversed_text = '<<'.join(reversed_parts) + '>>'
    return reversed_text

# Apply the function to reverse the word order of each question
dataset = dataset.map(lambda example: {
    'original_question': example['question'],
    'reversed_question': reverse_question(example['question']),
    'answer': example['answer']
})

# Display the first few examples to confirm the structure
print(dataset[:5])
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
