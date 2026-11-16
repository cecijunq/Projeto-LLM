# Projeto: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
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

## Artigos da revisão (5)
[2008.06239v2] Language Models as Few-Shot Learner for Task-Oriented Dialogue Systems
This paper explores the use of language models as few-shot learners in task-oriented dialogue systems. The study focuses on how few-shot learning can be applied to train modules like Natural Language Understanding (NLU), Dialogue State Tracking (DST), Dialogue Policy (DP), and Natural Language Generation (NLG) with minimal labeled data. The authors demonstrate that language models such as GPT-2 and GPT-3 can perform few-shot learning by providing a few examples as prompts. This approach reduces the need for extensive fine-tuning and parameter tuning for each task, making it a more efficient method for training dialogue systems with limited data. The results show that leveraging few-shot learning with language models improves the performance of dialogue systems in task-oriented settings.
[2507.20019v1] Anomaly Detection in Human Language via Meta-Learning: A Few-Shot Approach
This paper proposes a meta-learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language, such as spam, fake news, and hate speech, are challenging due to their sparsity and variability. The authors treat anomaly detection as a few-shot binary classification problem and leverage meta-learning to train models that generalize across tasks. Using datasets from domains like SMS spam, COVID-19 fake news, and hate speech, the model is evaluated on unseen tasks with minimal labeled anomalies. The method combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results demonstrate that the proposed method outperforms strong baselines in terms of F1 and AUC scores. Additionally, the paper highlights the robustness of the approach in handling diverse and unseen anomaly types, making it a valuable contribution to the field of few-shot learning in natural language processing.
[2311.14544v1] Inferring Latent Class Statistics from Text for Robust Visual Few-Shot Learning
This paper introduces a novel approach that leverages text-derived statistics to predict the mean and covariance of the visual feature distribution for each class. The study addresses the limitations of foundation models like CLIP in cross-domain robustness, especially in few-shot settings. By incorporating text as a modality to infer latent class statistics, the proposed method enriches the latent space, leading to more robust and generalizable few-shot learning models. The paper demonstrates that incorporating both mean and covariance statistics significantly improves the performance of few-shot learning models, particularly in scenarios where cross-domain robustness is crucial. This work provides a valuable contribution to the field of few-shot learning by leveraging textual information to enhance visual representation learning.
[2008.06239v2] Language Models as Few-Shot Learner for Task-Oriented Dialogue Systems
This paper explores the use of language models as few-shot learners in task-oriented dialogue systems. The study focuses on how few-shot learning can be applied to train modules like Natural Language Understanding (NLU), Dialogue State Tracking (DST), Dialogue Policy (DP), and Natural Language Generation (NLG) with minimal labeled data. The authors demonstrate that language models such as GPT-2 and GPT-3 can perform few-shot learning by providing a few examples as prompts. This approach reduces the need for extensive fine-tuning and parameter tuning for each task, making it a more efficient method for training dialogue systems with limited data. The results show that leveraging few-shot learning with language models improves the performance of dialogue systems in task-oriented settings.
[2507.20019v1] Anomaly Detection in Human Language via Meta-Learning: A Few-Shot Approach
This paper proposes a meta-learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language, such as spam, fake news, and hate speech, are challenging due to their sparsity and variability. The authors treat anomaly detection as a few-shot binary classification problem and leverage meta-learning to train models that generalize across tasks. Using datasets from domains like SMS spam, COVID-19 fake news, and hate speech, the model is evaluated on unseen tasks with minimal labeled anomalies. The method combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results demonstrate that the proposed method outperforms strong baselines in terms of F1 and AUC scores. Additionally, the paper highlights the robustness of the approach in handling diverse and unseen anomaly types, making it a valuable contribution to the field of few-shot learning in natural language processing.

## Plano
1. Load the SST-2 dataset and split it into training and test sets.
2. Select 10 examples (5 positive, 5 negative) for few-shot learning.
3. Generate multiple permutations of the 10 examples to test the impact of order.
4. For each permutation, format the examples as a prompt and use the few-shot examples to classify the test set using the local language model served by Ollama.
5. Measure the classification accuracy for each permutation and analyze the effect of example order on model performance.
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
