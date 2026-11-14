# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- running_experiments: outro_erro
- running_experiments: erro_de_execucao
- running_experiments: erro_de_sintaxe
- running_experiments: erro_de_execucao
- running_experiments: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2602.10661v2] This paper evaluates the performance of transformer-based language models on split-ergative case alignment in Georgian, a language with a rare grammatical case system. The study focuses on subject and object marking through permutations of nominative, ergative, and dative noun forms. A treebank-based approach is used to generate minimal pairs using the Grew query language, resulting in a dataset of 370 syntactic tests across seven tasks. Five encoder-only and two decoder-only models are evaluated using word- and sentence-level accuracy metrics. The results show that models perform worst in assigning the ergative case correctly, indicating challenges in syntactic processing for such grammatical structures.
[2308.11483v1] This paper investigates the sensitivity of Large Language Models (LLMs) to the order of options in multiple-choice questions. The study shows that LLMs exhibit a considerable performance gap, ranging from 13% to 75% in different benchmarks, when answer options are reordered, even when the content of the options remains the same. This finding highlights the challenges in fairly assessing LLMs, as their performance is influenced by the arrangement of options in multiple-choice tasks. The results emphasize the need for further research into improving the robustness of LLMs against such ordering variations, which is crucial for tasks like reasoning and fact-retrieving.
[2210.12187v2] This paper investigates whether neural models' estimates of syntactic surprisal can predict human processing difficulty from syntactic ambiguities. Humans exhibit garden path effects, slowing down when sentences are disambiguated in favor of less preferred structures. Surprisal theory suggests these slowdowns result from unpredictable words, but van Schijndel & Linzen (2021) found that language models underestimate human garden path effects. This work explores if this underestimation stems from humans prioritizing syntactic factors more than models do. The study proposes a method to evaluate how well neural models capture human syntactic sensitivity, providing insights into the limitations of current models in capturing human processing dynamics.
[2308.11483v1] This paper investigates the sensitivity of Large Language Models (LLMs) to the order of options in multiple-choice questions. The study demonstrates that LLMs exhibit a significant performance gap, ranging from 13% to 75% in different benchmarks, when answer options are reordered, even though the content of the options remains unchanged. This finding highlights the challenges in fairly assessing LLMs, as their performance is heavily influenced by the arrangement of options in multiple-choice tasks. The results emphasize the need for further research into improving the robustness of LLMs against such ordering variations, which is crucial for tasks like reasoning and fact-retrieving.
[2308.11483v1] This paper investigates the sensitivity of Large Language Models (LLMs) to the order of options in multiple-choice questions. The study shows that LLMs exhibit a considerable performance gap, ranging from 13% to 75% in different benchmarks, when answer options are reordered, even when the content of the options remains the same. This finding highlights the challenges in fairly assessing LLMs, as their performance is influenced by the arrangement of options in multiple-choice tasks. The results emphasize the need for further research into improving the robustness of LLMs against such ordering variations, which is crucial for tasks like reasoning and fact-retrieving.

## Plano
Evaluate the sensitivity of a local language model to word order in multiple-choice questions by comparing the model's accuracy on two versions of the same question: one with the correct answer first and one with the correct answer last. Use a test set of at least 200 examples, where each example consists of a question and four answer choices. For each question, generate two versions: one with the correct answer as the first option and one with the correct answer as the last option. Evaluate the model's accuracy on both versions using the same metric (e.g., percentage of correct answers). Compare the accuracy scores to determine if the model is sensitive to word order in multiple-choice questions.

## Código de preparação de dados
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('WaltonFuture/Diabetes-multiple-choice-questions', split='train[:200]')

# Check column names and a sample
print("Column names:", dataset.column_names)
print("Sample:", dataset[0])

# Extract the question and answer
questions = [example['problem'] for example in dataset]
answers = [example['answer'] for example in dataset]

# Prepare the data for the plan
print("Questions:", questions)
print("Answers:", answers)
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
