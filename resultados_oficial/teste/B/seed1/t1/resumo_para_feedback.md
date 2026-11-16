# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- report_writing: sem_relatorio

## Artigos da revisão (4)
[2308.11483v1] Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions. This paper investigates how large language models (LLMs) are sensitive to the order of answer options in multiple-choice questions. The study demonstrates that reordering the options can lead to a significant performance gap, ranging from 13% to 75% across different benchmarks. This sensitivity highlights challenges in fairly assessing LLMs' reasoning and fact-retrieval capabilities, as the task of multiple-choice questions is commonly used for these evaluations. The findings emphasize the need for more robust and order-invariant evaluation methods for LLMs.
[2608.05850v2] MameLoshnLM: Yiddish Language Model and Evaluation Benchmark. This paper introduces MameLoshnLM, the first open-source 8B-parameter language model specifically designed for Yiddish. The study highlights the challenges in Yiddish language modeling due to its limited digital presence and the scarcity of reliable evaluation resources. To address these issues, the authors present Oytser, a high-quality Yiddish pretraining corpus combining contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark for translation, linguistic analysis, information extraction, and language understanding. These resources are used to continue pretraining Llama 3.1 8B on Yiddish, aiming to improve the model's performance and benchmarking capabilities for the Yiddish language.
[2608.05850v2] MameLoshnLM: Yiddish Language Model and Evaluation Benchmark. This paper introduces MameLoshnLM, the first open-source 8B-parameter language model specifically designed for Yiddish. The study highlights the challenges in Yiddish language modeling due to its limited digital presence and the scarcity of reliable evaluation resources. To address these issues, the authors present Oytser, a high-quality Yiddish pretraining corpus combining contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark for translation, linguistic analysis, information extraction, and language understanding. These resources are used to continue pretraining Llama 3.1 8B on Yiddish, aiming to improve the model's performance and benchmarking capabilities for the Yiddish language.
[2608.05850v2] MameLoshnLM: Yiddish Language Model and Evaluation Benchmark. This paper introduces MameLoshnLM, the first open-source 8B-parameter language model specifically designed for Yiddish. The study highlights the challenges in Yiddish language modeling due to its limited digital presence and the scarcity of reliable evaluation resources. To address these issues, the authors present Oytser, a high-quality Yiddish pretraining corpus combining contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark for translation, linguistic analysis, information extraction, and language understanding. These resources are used to continue pretraining Llama 3.1 8B on Yiddish, aiming to improve the model's performance and benchmarking capabilities for the Yiddish language.

## Plano
The experiment will investigate whether language models are sensitive to word order in multiple-choice benchmarks. We will use the MMLU dataset, selecting a small subset of questions (e.g., 100 questions across different subjects). For each question, we will rephrase the answer options by altering the word order while preserving the semantic meaning. We will then evaluate the model's performance on both the original and rephrased options using the Ollama-served Llama 3.1 8B model. We will measure accuracy and compare the results to determine if there is a significant difference in performance due to word order changes. This experiment will be conducted in a few minutes on a single GPU, using only the local language model and minimal data.
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
