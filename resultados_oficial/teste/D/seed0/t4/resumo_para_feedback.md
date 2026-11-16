# Projeto: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: timeout
- data_preparation: erro_de_execucao
- data_preparation: timeout
- data_preparation: erro_de_execucao
- data_preparation: timeout
- data_preparation: erro_de_execucao
- data_preparation: timeout
- data_preparation: erro_de_execucao
- data_preparation: timeout
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- data_preparation: erro_de_execucao
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2205.15014v1] This paper introduces the Task-Prior Conditional Variational Auto-Encoder (TP-VAE) for few-shot image classification, focusing on nonuniform few-shot learning scenarios. The model is conditioned on support shots and constrained by a task-level prior regularization, achieving high performance in challenging nonuniform few-shot settings. It demonstrates significant improvements in accuracy, especially for 1-shot learning, and outperforms existing state-of-the-art methods across various standard few-shot image classification benchmarks.
[2311.14544v1] This paper introduces a novel approach called "Inferring Latent Class Statistics from Text for Robust Visual Few-Shot Learning." The method leverages text-derived statistics to predict the mean and covariance of the visual feature distribution for each class, enriching the latent space and improving the robustness and generalizability of few-shot learning models. By incorporating both mean and covariance statistics, the paper demonstrates significant improvements in few-shot learning performance, particularly in cross-domain scenarios. The proposed framework is evaluated on various benchmarks and shows promising results in enhancing the robustness of foundation models like CLIP in few-shot settings.
[2008.06239v2] This paper explores the application of language models as few-shot learners in task-oriented dialogue systems. It discusses how language models like GPT-2 and GPT-3 can be utilized for few-shot learning by priming the model with a small number of examples. The paper emphasizes the potential of language models to perform dialogue tasks with minimal fine-tuning, making them highly efficient for scenarios where data collection is costly. The study provides insights into how the order of examples might influence the model's ability to generalize and perform accurate text classification in few-shot settings, offering a foundation for further research on the impact of example order in language models.
[2507.20019v1] This paper proposes a meta-learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language, such as spam, fake news, and hate speech, are challenging due to their sparsity and variability. The paper treats anomaly detection as a few-shot binary classification problem and leverages meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, the method evaluates model generalization on unseen tasks with minimal labeled anomalies. The framework combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that the method outperforms strong baselines in F1 and AUC scores. The paper also highlights the potential of meta-learning in addressing the challenges of anomaly detection in human language.
[2205.15014v1] This paper introduces the Task-Prior Conditional Variational Auto-Encoder (TP-VAE) for few-shot image classification, focusing on nonuniform few-shot learning scenarios. The model is conditioned on support shots and constrained by a task-level prior regularization, achieving high performance in challenging nonuniform few-shot settings. It demonstrates significant improvements in accuracy, especially for 1-shot learning, and outperforms existing state-of-the-art methods across various standard few-shot image classification benchmarks.

## Plano
Does the order of few-shot examples change the accuracy of a language model on text classification?

Plan:
1. Use Ollama's local model (e.g., llama3) for all conditions.
2. Use a test set of 200 text classification examples, with a balanced distribution across classes (e.g., 50 examples per class).
3. For each example, create a few-shot prompt with 5 examples per class (e.g., 5 examples for class A, 5 for class B, etc.).
4. Compare the model's accuracy across the following conditions:
   - Random order of examples.
   - Ascending order of class labels.
   - Descending order of class labels.
   - No few-shot examples (control condition).
5. Evaluate the model's accuracy using the same metric (e.g., accuracy, F1 score) across all conditions.
6. Record the accuracy for each condition and compare the results to determine if the order of few-shot examples significantly affects the model's performance.

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
