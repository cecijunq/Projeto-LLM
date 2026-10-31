# Projeto: Do language models exhibit cognitive biases, such as confirmation bias or anchoring bias?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: formato_invalido
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- data_preparation: dependencia_ausente
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2406.02050v4] The paper introduces the Japanese Bias Benchmark (JBBQ) for analyzing social biases in Japanese large language models (LLMs). The study builds upon the English bias benchmark BBQ to evaluate social biases in Japanese LLMs. Results show that while Japanese LLMs with more parameters achieve higher accuracy on JBBQ, their bias scores also increase. Additionally, the paper explores methods like prompt-based bias warnings and chain-of-thought prompting, which can reduce the impact of biases in model outputs. However, further improvements are needed in extracting correct evidence to mitigate biases effectively.
[2109.14047v1] The paper introduces the Second Order WinoBias (SoWinoBias) test set, designed to measure latent gender bias in coreference resolution systems. The study highlights the presence of gender-induced bias in downstream tasks, even when explicit gender words are absent. The SoWinoBias dataset is used to evaluate current debiasing methods, focusing on how these methods affect embedding space properties and model performance. The results emphasize the challenges of detecting and mitigating subtle, latent forms of bias in language models.
[2410.16927v1] This paper investigates the presence of biases in AI-generated candidate interview reports by large language models (LLMs) such as Claude 3.5 Sonnet, GPT-4o, Gemini 1.5, and Llama 3.1 405B. The study evaluates biases related to gender, race, and age in these reports and assesses the effectiveness of LLM-based anonymization techniques in reducing such biases. Results show that anonymization can reduce certain biases, particularly gender bias, but the effectiveness varies across models and bias types. Llama 3.1 405B exhibited the lowest overall bias. The study also proposes a novel methodology for assessing inherent biases in LLMs by comparing anonymized and non-anonymized data, which has broader implications beyond recruitment applications.
[2104.00170v4] This paper critically examines the effectiveness of bias mitigation techniques in deep learning systems. It highlights that while various algorithms have been developed to mitigate bias, their effectiveness remains unclear due to inconsistent evaluation protocols, limited testing on diverse bias types, and potential overfitting to specific datasets. To address these issues, the authors introduce an improved evaluation protocol, sensible metrics, and a new dataset that enables a more rigorous assessment of bias mitigation algorithms. The study evaluates seven state-of-the-art techniques using a standardized network architecture and hyperparameter selection policy, providing insights into their strengths and limitations in reducing biases in deep learning models.
[2503.06054v1] This paper presents a detection framework to identify nuanced biases in Large Language Models (LLMs). The study highlights the challenges of detecting subtle biases that can propagate misinformation, influence decision-making, and reinforce stereotypes. The proposed framework integrates contextual analysis, interpretability through attention mechanisms, and counterfactual data generation to enhance bias detection. The results demonstrate the effectiveness of the framework in identifying both explicit and implicit biases in LLM outputs, offering a more comprehensive approach to bias detection and mitigation in AI systems.

## Plano
Step 1: Create a small dataset of 20-30 examples. Each example will consist of a factual statement (e.g., "The Earth orbits the Sun"), a biased prompt (e.g., "Do you think the Earth orbits the Sun?"), and a neutral prompt (e.g., "What is the correct scientific explanation for the Earth's movement around the Sun?").

Step 2: Load the Ollama local language model (e.g., 'llama3') and use its OpenAI-compatible API to generate responses for both the biased and neutral prompts for each example.

Step 3: Record the model's responses and analyze them qualitatively to determine if the model's answers are influenced by the prompt type. Look for evidence of confirmation bias (e.g., the model reinforces a pre-existing belief) or anchoring bias (e.g., the model relies heavily on the first piece of information provided).

Step 4: Compare the responses across all examples to identify patterns in how the model responds to biased vs. neutral prompts. Use this analysis to determine whether the model exhibits cognitive biases such as confirmation or anchoring bias.

Step 5: Summarize the findings and discuss the implications of the results for the use of language models in applications requiring unbiased reasoning.

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
