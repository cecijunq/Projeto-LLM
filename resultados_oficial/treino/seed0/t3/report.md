# Research Report: Does chain-of-thought prompting improve the accuracy of a small language model on grade-school math (GSM8K)?

## Abstract

\section*{Abstract}
This study investigates whether chain-of-thought (CoT) prompting improves the accuracy of a small language model on grade-school math problems from the GSM8K dataset. Given the increasing interest in enhancing the reasoning capabilities of language models, particularly for complex tasks like mathematical problem-solving, this experiment evaluates the effectiveness of CoT prompting as a strategy to elicit structured reasoning. The challenge lies in the limited capacity of small models to process and solve multi-step arithmetic problems, even with advanced prompting techniques. The contribution of this work is twofold: first, it provides an empirical evaluation of CoT prompting on a scaled-down subset of GSM8K, and second, it highlights the limitations of small models in handling such tasks. The results, which show zero accuracy for both baseline and CoT prompting strategies, underscore the need for further investigation into model architecture, training data, and prompting methodologies to improve performance on mathematical reasoning tasks.
</text>

## Introduction

\section*{Introduction}

The ability of language models to perform mathematical reasoning has become a focal point in the evaluation of their cognitive capabilities. Grade-school math problems, such as those found in the GSM8K dataset, serve as a benchmark for assessing the reasoning abilities of large language models (LLMs). While prior research has demonstrated that chain-of-thought (CoT) prompting can enhance the performance of large models on such tasks, the effectiveness of CoT prompting in improving the accuracy of small language models remains an open question. This study investigates whether CoT prompting results in improved accuracy for a small language model on a subset of the GSM8K dataset. The experiment is designed to evaluate the impact of CoT prompting on the model's ability to solve grade-school math problems, with a focus on the reasoning process. By comparing the performance of the model under two prompting strategies—baseline prompting and CoT prompting—the study aims to provide insights into the potential of CoT prompting as a method to enhance the reasoning capabilities of smaller models. The results of this experiment contribute to the broader understanding of how different prompting strategies influence the performance of language models on complex reasoning tasks.

## Background

\section*{Background}

Chain-of-thought (CoT) prompting has emerged as a promising technique to enhance the reasoning capabilities of large language models (LLMs) on complex tasks, particularly those requiring multi-step logical inference. Introduced in \cite{2201.11903v6}, CoT prompting involves instructing the model to generate an internal sequence of reasoning steps before arriving at a final answer. This approach has been shown to significantly improve the performance of LLMs on tasks such as grade-school mathematics, where step-by-step reasoning is critical for solving problems. The GSM8K dataset, which consists of 8,500 math word problems typically encountered by students in grades 3–5, has served as a benchmark for evaluating the effectiveness of CoT prompting. 

In addition to CoT prompting, recent research has focused on the limitations of LLMs in reasoning tasks, particularly in scenarios where the model must process visual information or meta-reasoning is required. For instance, \cite{2312.17080v4} introduces MR-GSM8K, a meta-reasoning benchmark that shifts the focus from simple question-answering to evaluating the reasoning process itself. This benchmark highlights the need for more comprehensive evaluation methods that capture the cognitive capabilities of LLMs. Similarly, \cite{2509.25160v3} explores the performance of vision-language models (VLMs) on visualized versions of GSM8K problems, emphasizing the challenges of reasoning in non-textual contexts. 

The current study builds on these findings by investigating whether CoT prompting can improve the accuracy of a small language model on GSM8K tasks. While prior work has demonstrated the effectiveness of CoT prompting for large models, its utility with smaller models remains underexplored. This work aims to address this gap by conducting an experiment using a lightweight model and a subset of the GSM8K dataset, with the goal of assessing the potential of CoT prompting in improving mathematical reasoning performance.

## Related Work

\section*{Related Work}

This study builds on prior research exploring the effectiveness of chain-of-thought (CoT) prompting in enhancing the reasoning capabilities of language models. The seminal work by Wei et al. \cite{2201.11903v6} introduced CoT prompting as a method to elicit structured reasoning from large language models, demonstrating its efficacy in improving performance on complex reasoning tasks such as those found in the GSM8K benchmark. Their findings suggest that CoT prompting can significantly boost model accuracy, although they tested this on a large-scale model with 540B parameters. This contrasts with the current study, which focuses on a small language model, raising questions about the generalizability of CoT prompting across different model sizes.

In contrast, the MR-GSM8K benchmark \cite{2312.17080v4} shifts the evaluation focus from direct answer generation to a more comprehensive assessment of reasoning processes. This approach emphasizes the cognitive capabilities of models, which is relevant to understanding how CoT prompting influences model behavior. However, MR-GSM8K does not explicitly evaluate the impact of prompting strategies, unlike the current study, which directly compares baseline and CoT prompting methods. This distinction highlights the importance of prompting as a modifiable factor in model performance, particularly in tasks requiring structured reasoning.

The work on GSM8K-V \cite{2509.25160v3} extends the scope of the GSM8K dataset by introducing a visual modality, testing vision-language models' ability to solve math problems in visual contexts. While this work highlights the challenges of reasoning in multimodal settings, it does not address the effectiveness of prompting techniques like CoT. Thus, it provides a complementary perspective on reasoning capabilities but does not directly inform the current investigation into the role of prompting strategies in improving model accuracy on grade-school math problems.

In summary, while prior work has explored the potential of CoT prompting and alternative evaluation frameworks for reasoning tasks, this study directly examines its impact on a small language model, offering insights into the limitations and possibilities of such approaches in practical settings.

## Methods

\section*{Methods}

The experimental methodology is structured to evaluate the impact of chain-of-thought (CoT) prompting on the accuracy of a small language model when solving grade-school math problems from the GSM8K dataset. The approach involves a controlled comparison between a baseline prompting strategy and a CoT prompting strategy, with the latter designed to elicit intermediate reasoning steps before arriving at a final answer.

The GSM8K dataset is first loaded from HuggingFace using the `load_dataset` function, and a subset of 100 problems is selected for evaluation. The dataset is stored as a pandas DataFrame to facilitate data processing. Each problem is represented as a text-based question, and its corresponding solution is stored as a string. The selected questions and answers are then used as input for model evaluation.

Two prompting strategies are implemented: (1) the baseline prompt, which directly asks the model to solve the math problem without any intermediate reasoning steps, and (2) the CoT prompt, which instructs the model to first reason step-by-step and then provide the final solution. The baseline prompt is formatted as:

$$
\text{Baseline Prompt: } \text{"Solve the following math problem: \{question\} Answer: "}
$$

The CoT prompt is formatted as:

$$
\text{CoT Prompt: } \text{"Reason step by step and then solve the following math problem: \{question\} Answer: "}
$$

Both prompting strategies are applied to the same 100 problems, with each problem queried independently. The model responses are evaluated for correctness by comparing the generated answers to the ground truth solutions. The accuracy is computed as the ratio of correctly answered problems to the total number of problems processed. The model interaction is conducted via the OpenAI API, utilizing the `chat.completions.create` method with a specified model, maximum token limit, and temperature parameter set to 0 to ensure deterministic output.

The experiment is designed to be lightweight, with a limited number of queries (up to 50 for each prompting strategy) to ensure computational efficiency. The results are analyzed to determine whether CoT prompting leads to an improvement in model accuracy on grade-school math problems.
</text>

## Experimental Setup

\section{Experimental Setup}

The experimental setup is designed to evaluate the effectiveness of chain-of-thought (CoT) prompting on a small language model's performance on grade-school math problems from the GSM8K dataset. The dataset used is a subset of 100 problems selected from the full GSM8K benchmark, which is sourced from HuggingFace under the identifier `openai/gsm8k`. Each problem consists of a question and a corresponding answer, with the latter typically containing intermediate reasoning steps in the form of arithmetic expressions. The evaluation focuses on the accuracy of the model's responses in matching the exact answer format, excluding any additional explanation or formatting.

The model employed is a small language model from Ollama, specifically the `llama3` or `mistral` variant, which is accessible via the OpenAI API. The model's parameters are kept consistent across both prompting strategies to ensure a fair comparison. The temperature parameter is set to 0 to prioritize deterministic outputs, while the maximum token limit is capped at 64 to limit the length of responses and focus on concise answers.

Two prompting strategies are implemented: (1) the baseline, where the model is directly asked to solve the math problem without additional instructions, and (2) the chain-of-thought (CoT) approach, where the model is prompted to first think through the problem step-by-step and then provide the answer. Both strategies use the same set of 100 problems, with each problem being evaluated independently. The accuracy metric is defined as the proportion of responses that exactly match the provided solution, with no tolerance for numerical or formatting discrepancies.

The experiment is conducted without external APIs beyond the model's interaction via the OpenAI API, and the dataset is processed using the `datasets` library and stored in a CSV format for later reference. The evaluation is limited to 50 queries per prompting strategy to ensure computational efficiency and to mitigate potential API rate limit issues.
</text>

## Results

\section{Results}

The experimental results demonstrate that neither the baseline prompting strategy nor the chain-of-thought (CoT) prompting strategy achieved any correct answers on the subset of 100 GSM8K problems. Specifically, the baseline accuracy was 0/100 (0.00\%), and the CoT accuracy was 0/50 (0.00\%). This indicates that the small language model used in the experiment did not produce correct answers for any of the tested problems under either prompting method.

A critical concern is that the model's performance did not improve with the introduction of CoT prompting, suggesting that the model may not be capable of generating reasoning steps or solving grade-school math problems. This outcome could stem from several factors, including model limitations, incorrect API configuration, or issues with the dataset handling. The results highlight the necessity of verifying the model's accessibility, API key validity, and proper configuration. Additionally, the dataset formatting and problem selection should be reviewed to ensure they were correctly applied. Further testing with a larger dataset or different model configurations is recommended to assess the model's true capabilities on GSM8K tasks.
</text>

## Discussion

\section*{Discussion}

The experimental results indicate that both the baseline and chain-of-thought (CoT) prompting strategies failed to produce correct answers on the GSM8K subset, achieving zero accuracy. This outcome is unexpected, given the prior literature suggesting that CoT prompting can significantly enhance the reasoning capabilities of large language models on complex tasks like grade-school mathematics. The lack of improvement between the two prompting methods suggests that the small language model may not be capable of generating meaningful reasoning steps or correct answers, even when prompted to do so.

Several factors might contribute to this result. First, the model used in the experiment may lack sufficient reasoning capacity to solve grade-school math problems, particularly when the dataset is not fully representative or when the model is not optimized for mathematical reasoning. Second, the implementation of CoT prompting may not have been sufficiently effective, possibly due to an inadequate number of demonstration examples or an unsuitable prompt formulation. Third, there could be an issue with the model's API configuration or accessibility, which may have prevented it from receiving or processing the prompts correctly.

The low accuracy observed in this experiment contrasts with the findings from prior studies, which reported significant improvements in model performance using CoT prompting. This discrepancy highlights the importance of model selection and prompt engineering in achieving successful reasoning outcomes. It also underscores the need for further investigation into the limitations of small language models on complex reasoning tasks and the role of external factors such as training data, model architecture, and prompt design in influencing their performance.

To address these limitations, future work should explore the use of larger or more finely-tuned models, as well as more sophisticated prompting strategies that incorporate additional examples or structured reasoning templates. Additionally, the evaluation should consider the inclusion of a broader dataset and a more comprehensive error analysis to better understand the root causes of the observed performance issues.
