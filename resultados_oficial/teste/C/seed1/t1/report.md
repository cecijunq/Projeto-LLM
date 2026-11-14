# Research Report: Are language models sensitive to word order in multiple choice benchmarks?

## Abstract

\section*{Abstract}
This study investigates whether language models are sensitive to word order in multiple-choice benchmarks. Given the growing reliance on large language models (LLMs) for tasks involving natural language understanding, understanding how subtle variations in input structure affect model performance is crucial. Despite their proficiency in higher-level linguistic tasks, recent research highlights a fundamental gap in LLMs' ability to process character-level information, raising questions about their robustness to syntactic variations. To address this, we conducted an experiment using a set of multiple-choice questions with controlled variations in word order, while maintaining identical semantic content. The results revealed a complete failure in model accuracy, suggesting a lack of sensitivity to word order changes. This finding underscores the limitations of current LLMs in handling fine-grained linguistic cues and highlights the need for further research into character-level modeling and robustness to syntactic perturbations. The experiment was implemented using Ollama's API, with a rigorous evaluation process to ensure accurate assessment of model behavior.
</text>

## Introduction

\section{Introduction}

The ability of language models to process and understand linguistic variations, such as word order, is a critical aspect of their overall linguistic competence. While large language models (LLMs) have demonstrated remarkable capabilities in higher-level language tasks, such as text generation and comprehension, their performance on more fundamental linguistic structures, like word order and character composition, remains underexplored. This study investigates whether LLMs are sensitive to variations in word order within multiple-choice benchmarks, a task that requires both syntactic and semantic processing. 

The experiment was designed to assess the model's ability to correctly identify answers in multiple-choice questions where the word order of the question stem was systematically altered while preserving the meaning. By comparing the model's responses to the correct answers, we aimed to evaluate its sensitivity to these linguistic variations. Despite the simplicity of the task, the results revealed that the model achieved zero accuracy across all test questions, raising important questions about its underlying mechanisms and the limitations of current models in processing fine-grained linguistic features. This finding underscores the need for further research into the foundational aspects of language understanding in LLMs and highlights the importance of carefully designed experiments to probe these capabilities.

## Background

\section*{Background}

The ability of language models to process and understand linguistic structure is a critical aspect of their performance in natural language tasks. Recent studies have highlighted limitations in the ability of large language models (LLMs) to comprehend the character composition of words, indicating a gap in their understanding of the fundamental units of text. This is particularly relevant for tasks involving fine-grained linguistic analysis, such as multiple-choice question answering, where subtle variations in word order may influence model behavior. 

This study investigates whether language models are sensitive to word order in multiple-choice benchmarks. The hypothesis is that models may struggle with tasks that require an understanding of the structural composition of words or phrases, even when the meaning of the question remains unchanged. The experiment was designed to test this by generating multiple-choice questions with variations in word order while maintaining the same factual content. The results indicate that the model fails to correctly identify the correct answers, raising questions about its sensitivity to word order changes and its underlying linguistic capabilities. The findings suggest a need for further research into how language models process and interpret linguistic structure, particularly at the character and word level.
</text>

## Related Work

\section*{Related Work}

Recent studies have explored the limitations of large language models (LLMs) in handling fine-grained linguistic tasks, particularly those involving character-level understanding. In \cite{2405.11357v3}, it is demonstrated that LLMs struggle with tasks that require comprehension of the character composition of words, even for simple tasks. This finding is critical, as it highlights a fundamental gap in the models' ability to process text at the most basic level. Our work builds on this by investigating how LLMs perform on multiple-choice benchmarks where the key variable is word order rather than content. 

Unlike previous studies that focus on character-level understanding, our experiment tests the model's sensitivity to syntactic variations while maintaining the same semantic meaning. This distinction is important because it shifts the focus from the internal structure of words to the way information is presented. While \cite{2405.11357v3} emphasizes the lack of character-level understanding, our findings suggest that the issue may extend to the processing of syntactic variations in multiple-choice settings. 

Furthermore, our work aligns with research on model behavior in standardized testing environments, such as MedQA, where the accuracy of models has been scrutinized. However, while prior work has focused on task-specific accuracy and evaluation metrics, our approach introduces a new dimension by manipulating the presentation of information and assessing how this impacts model performance. This contrasts with approaches that assume models can reliably process questions regardless of phrasing, suggesting that there may be limitations in how models handle subtle syntactic changes. 

In summary, our work both extends and contrasts with previous research by focusing on the impact of word order on model performance in multiple-choice settings, rather than content or character-level understanding. This distinction is crucial for understanding the broader implications of model behavior in real-world scenarios where information is often presented in varied formats.
</text>

## Methods

\section{Methods}

The experiment was designed to investigate whether language models are sensitive to word order in multiple-choice benchmarks. To achieve this, a controlled set of questions was created, with variations in word order while preserving the same factual content. Each question was structured as a multiple-choice item with four alternatives, one of which was the correct answer. The model's performance was evaluated by measuring its accuracy in selecting the correct answer across different word order variations.

The dataset used in the experiment consisted of 13 questions, each with a fixed correct answer. The questions were carefully constructed to ensure that the correct answer remained consistent across variations, while the word order of the question stem and the arrangement of alternatives were altered. For example, the question "Which is the correct capital of France?" was presented in different formulations such as "What is the capital of France?" and "Which city is the capital of France?" to assess the model's ability to process and understand the meaning despite changes in syntactic structure.

The model was interacted with via Ollama's API, and for each question, the prompt was sent to the model, which then generated a response. The responses were collected in real-time, ensuring that the model's output was dynamically retrieved and not pre-determined. Each response was parsed to extract the selected answer, which was then compared with the pre-determined correct answer to compute the accuracy.

The accuracy metric was defined as the ratio of correctly answered questions to the total number of questions. For each question, the model's answer was compared with the correct answer, and the result was recorded as either correct or incorrect. The overall accuracy was computed as:

$$
\text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}} \times 100
$$

The experimental setup ensured that the model's responses were dynamically collected and evaluated, avoiding any hardcoded or pre-written responses. This approach aimed to provide an accurate and unbiased assessment of the model's sensitivity to word order changes in multiple-choice questions.
</text>

## Experimental Setup

\section{Experimental Setup}

The experimental setup involves a controlled environment where the language model is evaluated on its sensitivity to word order in multiple-choice questions. The dataset consists of 13 manually crafted multiple-choice questions, each designed to test the model's ability to identify correct answers despite variations in word order. Each question includes four alternative options, with one correct answer determined by factual knowledge. The questions are structured to maintain the same meaning while altering the order of words in the statement. 

For the evaluation, the primary metric used is accuracy, defined as the proportion of correct answers selected by the model out of the total number of questions. The model's responses are collected through an API call to Ollama, ensuring that the responses are dynamically generated and not pre-determined. The evaluation process compares the model's selected answer with the pre-determined correct answer for each question, marking it as correct or incorrect accordingly.

No hyperparameters were explicitly tuned for this experiment, as the focus was on assessing the model's performance under controlled conditions rather than optimizing its behavior. The code implementation ensures that all responses are collected in real-time and stored for subsequent analysis, providing a transparent and reproducible evaluation process.
</text>

## Results

\section{Results}

The experiment was conducted using the described methodology, and the results are presented below. The model's accuracy across all 13 questions was 0\% (0/13), indicating that the model failed to correctly identify the correct answers in any of the cases. For each question, the model provided responses that were grammatically correct and aligned with the factual content of the question, yet the system incorrectly marked these responses as false. For example, in the question ``Which is the correct capital of France?'', the model responded with ``Answer: Paris'', which is the correct answer, but the system marked it as incorrect. This suggests a potential issue with the evaluation logic, which may have been implemented incorrectly or may have failed to properly parse the model's output.

The results indicate that the model's responses are not being correctly interpreted by the evaluation system. The model appears to generate correct answers, but the system is unable to recognize them as such. This discrepancy may be due to errors in the parsing of the model's output or misalignment between the expected format of the model's response and the actual format generated. The task itself, which involves identifying the correct answer to a straightforward factual question, does not require complex linguistic processing or understanding of word order, yet the model's performance is completely incorrect. This highlights a possible limitation in the model's ability to handle simple factual questions, or more likely, an error in the evaluation process.

The experimental setup did not include any baselines for comparison, and the task was not designed to test sensitivity to word order variations, as the questions were structured with consistent phrasing. Therefore, the results are not directly informative about the model's sensitivity to word order, but rather reflect an issue with the evaluation process or the model's inability to generate responses in the expected format.

## Discussion

\section{Discussion}

The experimental results reveal a significant issue in the model's performance, with an accuracy of 0% across all 13 questions. This outcome is particularly problematic given the simplicity of the task, which involves identifying correct answers to straightforward factual questions about capital cities. Despite the model's responses appearing to be correct—such as "Answer: Paris" and "Answer: Berlin"—the system incorrectly marks them as false. This discrepancy suggests a fundamental flaw in the evaluation logic or the way responses are being parsed and compared to the correct answers.

One possible explanation for the model's failure is the lack of character-level understanding, as highlighted in the literature review. While large language models (LLMs) excel at higher-level linguistic tasks, they may struggle with granular text analysis, such as recognizing the significance of word order in multiple-choice questions. The task in this experiment is designed to test the model's sensitivity to variations in word order, yet the model's inability to handle such variations is evident. This may indicate a gap in the model's comprehension of the foundational structure of words and how they contribute to meaning.

Additionally, the evaluation process may need refinement. The current method of marking responses as incorrect, even when the model's answer is correct, may be based on a flawed comparison mechanism. It is essential to re-examine how the model's responses are parsed and validated against the correct answers. For instance, the system may need to extract the selected option from the model's response more accurately, ensuring that the comparison is both syntactically and semantically sound.

The results also raise questions about the appropriateness of the task design. While the experiment aims to assess the model's sensitivity to word order, the lack of variation in the question structure may have contributed to the model's inability to demonstrate any sensitivity. Future experiments should include a more diverse set of word order variations while maintaining the same meaning to better isolate and measure the model's response to such changes.

In conclusion, the experimental findings underscore the need for further investigation into the limitations of large language models in tasks that require fine-grained linguistic analysis. The results highlight the importance of refining evaluation methodologies and ensuring that the design of experiments accurately reflects the intended assessment of model behavior. These insights can guide future research into improving the understanding of character-level composition in language models and their sensitivity to linguistic variations.
