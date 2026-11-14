# Projeto: Are language models sensitive to word order in multiple choice benchmarks?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: busca_em_loop
- report_writing: sem_relatorio

## Artigos da revisão (4)
[2308.11483v1] Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions
This paper investigates the sensitivity of Large Language Models (LLMs) to the order of options in multiple-choice questions. The study highlights that LLMs exhibit considerable performance variability when the answer options are reordered, even when the content of the options remains the same. The authors demonstrate a performance gap of approximately 13% to 75% across different benchmarks, indicating that the order of options significantly impacts model performance. This finding underscores the importance of understanding and addressing the limitations of LLMs in tasks that rely on reasoning and fact-retrieval capabilities. The results emphasize the need for further research into improving the robustness of LLMs against such ordering effects.
[2608.05850v2] MameLoshnLM: Yiddish Language Model and Evaluation Benchmark
This paper introduces MameLoshnLM, the first open-source 8B-parameter language model specifically designed for Yiddish. The study highlights the challenges in Yiddish language modeling, including the limited digital presence of Yiddish and the scarcity of reliable evaluation resources. Existing multilingual corpora and benchmarks are often inadequate, containing noisy, machine-translated, and misclassified text. To address these issues, the authors present Oytser, a high-quality Yiddish pretraining corpus combining contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark for translation, linguistic analysis, information extraction, and language understanding. These resources enable the continued pretraining of Llama 3.1 8B on Yiddish, contributing to the advancement of Yiddish language modeling.
[2308.11483v1] Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions
This paper investigates the sensitivity of Large Language Models (LLMs) to the order of options in multiple-choice questions. The study highlights that LLMs exhibit considerable performance variability when the answer options are reordered, even when the content of the options remains the same. The authors demonstrate a performance gap of approximately 13% to 75% across different benchmarks, indicating that the order of options significantly impacts model performance. This finding underscores the importance of understanding and addressing the limitations of LLMs in tasks that rely on reasoning and fact-retrieval capabilities. The results emphasize the need for further research into improving the robustness of LLMs against such ordering effects.
[2608.05850v2] MameLoshnLM: Yiddish Language Model and Evaluation Benchmark
This paper introduces MameLoshnLM, the first open-source 8B-parameter language model specifically designed for Yiddish. The study highlights the challenges in Yiddish language modeling, including the limited digital presence of Yiddish and the scarcity of reliable evaluation resources. Existing multilingual corpora and benchmarks are often inadequate, containing noisy, machine-translated, and misclassified text. To address these issues, the authors present Oytser, a high-quality Yiddish pretraining corpus combining contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark for translation, linguistic analysis, information extraction, and language understanding. These resources enable the continued pretraining of Llama 3.1 8B on Yiddish, contributing to the advancement of Yiddish language modeling.

## Plano
1. **Dataset Selection**: Use a small subset of the MMLU dataset (e.g., 100 questions from 5 subjects) to ensure the experiment runs quickly on a single GPU. Focus on multiple-choice questions with clear answer options.

2. **Question Preparation**: For each question, extract the answer options (A, B, C, D) and generate reordered versions (e.g., shuffle the order of the options while keeping the content unchanged). Ensure all variations maintain the same meaning and factual content.

3. **Model Interaction**: Use the Ollama API to interact with the local language model. For each question variation, prompt the model to select the correct answer. Record the model's response and whether it matches the correct option.

4. **Performance Evaluation**: Compare the model's accuracy across the original and reordered options. Calculate the accuracy rate for each condition and analyze whether there is a statistically significant difference in performance based on the order of the options.

5. **Result Analysis**: Summarize the findings, including the accuracy rates and any observed sensitivity to word order. Discuss how these results align with the findings from [2308.11483v1], and consider implications for model robustness and reliability in reasoning tasks.
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
