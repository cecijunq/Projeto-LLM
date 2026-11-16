# Projeto: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Etapas
- literature_review: FALHOU
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- literature_review: fase_falhou
- running_experiments: fase_falhou
- literature_review: busca_em_loop
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- running_experiments: outro_erro
- running_experiments: timeout
- report_writing: sem_relatorio

## Artigos da revisão (0)
(nenhum)

## Plano
{
  "objective": "Investigate whether the order of few-shot examples affects the accuracy of a language model on text classification tasks.",
  "data": {
    "source": "HuggingFace 'imdb' dataset",
    "subset_size": 20,
    "test_set_size": 10
  },
  "model": {
    "name": "llama2",
    "provider": "Ollama (OpenAI-compatible API)"
  },
  "experimental_design": {
    "version_a": {
      "description": "Few-shot examples ordered by label (positive first, then negative)",
      "prompt_template": "Classify the following text as positive or negative. Here are some examples:\n{examples}\nText: {text}\nLabel:"
    },
    "version_b": {
      "description": "Few-shot examples ordered randomly",
      "prompt_template": "Classify the following text as positive or negative. Here are some examples:\n{examples}\nText: {text}\nLabel:"
    }
  },
  "evaluation": {
    "metrics": ["accuracy"],
    "test_set": "Randomly selected 10 examples from the 'imdb' dataset"
  },
  "statistical_analysis": {
    "method": "t-test",
    "significance_level": 0.05
  },
  "execution": {
    "notes": "Ensure the experiment runs within a few minutes on a single GPU. Use a small dataset and limit the number of model queries to minimize computational load."
  }
}

## Código de preparação de dados
```python
from datasets import load_dataset

# Load the 'imdb' dataset with the specified split and columns
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Display the first few examples to confirm the data structure
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
