# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (2)
[2308.11483v1] Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions: This paper investigates how large language models (LLMs) are affected by the order of options in multiple-choice questions. The study reveals that LLMs exhibit significant sensitivity to the order of answer choices, leading to a performance gap of approximately 13% to 75% across different benchmarks when the options are reordered. This finding highlights the limitations of LLMs in handling tasks where the sequence of choices is critical, such as multiple-choice question answering. The results emphasize the need for further research into improving the robustness of LLMs against such ordering variations.
[2308.11483v1] Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions: This paper investigates how large language models (LLMs) are affected by the order of options in multiple-choice questions. The study reveals that LLMs exhibit significant sensitivity to the order of answer choices, leading to a performance gap of approximately 13% to 75% across different benchmarks when the options are reordered. This finding highlights the limitations of LLMs in handling tasks where the sequence of choices is critical, such as multiple-choice question answering. The results emphasize the need for further research into improving the robustness of LLMs against such ordering variations.

## Plano
1. **Dataset Selection**: Use the MultiRC dataset, which is a reading comprehension benchmark with multiple-choice questions. This dataset is suitable for evaluating model performance on answer choices.
2. **Reordering Setup**: For each question, generate multiple reordered versions of the answer choices (e.g., random permutations of the options).
3. **Model Interaction**: Use the local language model served by Ollama through an OpenAI-compatible API. For each question, prompt the model to answer the question using the original and reordered options.
4. **Performance Evaluation**: Measure the model's accuracy on both the original and reordered options. Compare the results to determine if there is a significant sensitivity to word order.
5. **Analysis**: Analyze the performance difference across the original and reordered options to assess the model's sensitivity to answer choice order.
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
