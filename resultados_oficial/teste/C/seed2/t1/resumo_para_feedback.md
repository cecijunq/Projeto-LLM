# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- report_writing: sem_relatorio

## Artigos da revisão (3)
[2308.11483v1] Large Language Models (LLMs) have demonstrated remarkable capabilities in various NLP tasks. However, previous works have shown these models are sensitive towards prompt wording, and few-shot demonstrations and their order, posing challenges to fair assessment of these models. As these models become more powerful, it becomes imperative to understand and address these limitations. In this paper, we focus on LLMs robustness on the task of multiple-choice questions -- commonly adopted task to study reasoning and fact-retrieving capability of LLMs. Investigating the sensitivity of LLMs towards the order of options in multiple-choice questions, we demonstrate a considerable performance gap of approximately 13% to 75% in LLMs on different benchmarks, when answer options are reordered, even when the content of the question and the correct answer remain unchanged. This sensitivity to option order suggests that LLMs may not be fully reliable in tasks where the order of choices is not controlled, highlighting the need for further research into improving robustness in such scenarios.
[2407.15018v2] Multiple-choice question answering (MCQA) is a key competence of performant transformer language models that is tested by mainstream benchmarks. However, recent evidence shows that models can have quite a range of performance, particularly when the task format is diversified slightly (such as by shuffling answer choice order). In this work we ask: how do successful models perform formatted MCQA? We employ vocabulary projection and activation patching methods to localize key hidden states that encode relevant information for predicting the correct answer. We find that the prediction of a specific answer symbol is causally attributed to a few middle layers, and specifically their multi-head self-attention mechanisms. We show that subsequent layers increase the probability of the predicted answer, suggesting that these layers refine and consolidate the initial predictions. This study provides insights into the internal mechanisms of language models during MCQA and highlights the impact of answer choice order on model performance.
[2407.15018v2] Multiple-choice question answering (MCQA) is a key competence of performant transformer language models that is tested by mainstream benchmarks. However, recent evidence shows that models can have quite a range of performance, particularly when the task format is diversified slightly (such as by shuffling answer choice order). In this work, we ask: how do successful models perform formatted MCQA? We employ vocabulary projection and activation patching methods to localize key hidden states that encode relevant information for predicting the correct answer. We find that the prediction of a specific answer symbol is causally attributed to a few middle layers, and specifically their multi-head self-attention mechanisms. We show that subsequent layers increase the probability of the predicted answer, suggesting that these layers refine and consolidate the initial predictions. This study provides insights into the internal mechanisms of language models during MCQA and highlights the impact of answer choice order on model performance.

## Plano
Evaluate the sensitivity of a language model to the order of answer choices in multiple-choice questions by comparing its performance on the original and shuffled versions of the same questions from the "MultipleChoice" dataset. Use the Ollama API to interact with the local language model. Split the dataset into two groups: one with the original option order and one with shuffled options. For each question, prompt the model to answer and record its accuracy. Compare the accuracy across the two conditions to assess sensitivity to word order.

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
