# Projeto: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: formato_invalido
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2008.06239v2] Language models can act as few-shot learners for task-oriented dialogue systems by being primed with a small number of examples. This paper investigates how language models like GPT-2 and GPT-3 can perform few-shot learning without requiring fine-tuning, which is a key insight for understanding how model behavior can be influenced by the order and presentation of examples. The study highlights the potential of language models to adapt to new tasks with minimal examples, which is directly relevant to the topic of whether the order of few-shot examples impacts classification accuracy. The paper also discusses the role of prompt engineering and example order in shaping model output, providing foundational insights for further research on this topic.
[2311.14544v1] This paper introduces a novel approach to enhance the robustness of few-shot learning models by leveraging text-derived statistics to predict the mean and covariance of the visual feature distribution for each class. The method enriches the latent space, leading to more generalizable and robust few-shot learning models. The paper demonstrates the effectiveness of incorporating both mean and covariance statistics in improving few-shot classification accuracy, especially in cross-domain scenarios. This work is relevant to the topic as it explores how additional information (like text) can influence the performance of few-shot learning models, which may indirectly relate to the impact of example order on model accuracy.
[2205.15014v1] This paper proposes the Task-Prior Conditional Variational Auto-Encoder (TP-VAE) model, which addresses the challenge of nonuniform few-shot learning scenarios where the number of samples per class is not equal. The study demonstrates that transductive methods outperform inductive methods in few-shot image classification tasks. TP-VAE is conditioned on support shots and constrained by a task-level prior regularization, leading to improved performance in challenging nonuniform few-shot settings. The method achieves a 3% increase in 1-shot classification accuracy compared to state-of-the-art approaches, highlighting the importance of task-level priors in enhancing model performance. While the paper focuses on image classification, its insights on the impact of data distribution and task-level regularizations are relevant to understanding how the order of few-shot examples might influence model accuracy in text classification tasks.
[2205.08157v1] The paper introduces an Uncertainty-Based Network for Few-shot Image Classification, which addresses the limitations of existing transductive methods by incorporating uncertainty modeling. The study highlights that traditional methods often optimize models based solely on classification scores without considering the uncertainty of these scores, leading to suboptimal performance. The proposed method uses mutual information to model the uncertainty of classification results, assigning weights to classification scores based on this uncertainty. This approach allows for iterative updates of the model, improving its adaptability to new tasks. The paper demonstrates that incorporating uncertainty into the training process enhances the robustness and accuracy of few-shot classification models, which is relevant to the topic as it explores how the integration of additional information (like uncertainty) can influence model performance, potentially affecting the impact of example order on classification accuracy.
[2507.20019v1] This paper proposes a meta-learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language, ranging from spam and fake news to hate speech, pose a major challenge due to their sparsity and variability. The paper treats anomaly detection as a few-shot learning problem, leveraging meta-learning to adapt to new anomaly types with minimal examples. The framework demonstrates how the order and structure of few-shot examples can influence the model's ability to generalize and detect anomalies, which is directly relevant to the research topic of whether the order of few-shot examples impacts classification accuracy. The study provides insights into the role of example presentation in shaping model behavior, particularly in the context of text classification tasks.

## Plano
We will design a simple experiment to investigate whether the order of few-shot examples impacts the accuracy of a language model on text classification. The experiment will use the Ollama language model via an OpenAI-compatible API and the HuggingFace dataset 'imdb' for text classification. The plan is as follows:

1. **Setup**: Load the 'imdb' dataset and split it into training and testing sets. Use only the testing set for evaluation.
2. **Few-shot Examples**: Select 5 examples for each class (positive and negative) from the training set.
3. **Vary Order**: Create multiple permutations of the few-shot examples for each class (e.g., 5 different orders).
4. **Prompt Construction**: Construct prompts for the language model by concatenating the few-shot examples in different orders, followed by the test sample.
5. **Model Inference**: Use the Ollama API to generate predictions for each test sample with different example orders.
6. **Accuracy Measurement**: Compute the classification accuracy for each order and compare across permutations.
7. **Statistical Analysis**: Use a t-test or ANOVA to determine whether the order of few-shot examples significantly impacts the model's accuracy.

The experiment will be run on a single GPU and should take a few minutes to complete. The data size is small, and only the local language model served by Ollama will be used.
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
