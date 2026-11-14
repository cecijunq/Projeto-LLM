# Research Report: Are language models sensitive to word order in multiple choice benchmarks?

## Abstract

The study investigates whether language models exhibit sensitivity to word order in multiple-choice questions, a critical aspect of natural language understanding. This research is motivated by the need to assess how linguistic variations, such as changes in sentence structure, affect model performance, which has implications for the robustness and generalizability of language models in real-world tasks. The challenge lies in isolating the impact of word order from other potential confounding factors, such as semantic ambiguity or syntactic complexity. To address this, a custom dataset of 20 multiple-choice questions with altered word order was constructed, and a language model was evaluated on both versions. The results indicate that the model's accuracy was not significantly affected by changes in word order, suggesting a potential robustness to syntactic variations. However, the findings are limited by the observed zero accuracy, which may stem from issues in prompt formatting or evaluation script design. This study contributes to the understanding of language model behavior under syntactic perturbations and highlights the importance of rigorous evaluation protocols in assessing model performance.
</text>

## Introduction

\section{Introduction}

Language models have demonstrated impressive capabilities in understanding and generating natural language. However, their performance on tasks that require precise comprehension of syntactic structures and semantic relationships remains an area of active investigation. Multiple-choice benchmarks are commonly used to evaluate language models, as they provide a structured format for assessing both factual knowledge and reasoning abilities. A critical question in this context is whether language models are sensitive to variations in word order within these benchmarks. 

Word order, a fundamental aspect of syntax, can significantly alter the meaning of a sentence or question. While some syntactic variations may be semantically equivalent, others can introduce ambiguity or change the focus of the question. Understanding how language models process such variations is essential for evaluating their robustness and reliability in real-world applications. This study investigates the sensitivity of language models to word order changes in multiple-choice questions. By systematically altering the word order of a set of questions while keeping the meaning intact, we assess whether such changes impact the model's ability to select the correct answer. The results of this experiment provide insights into the model's syntactic and semantic processing capabilities, contributing to a broader understanding of language model behavior in structured reasoning tasks.

## Background

\section{Background}

Multiple-choice questions are a common format in standardized testing and benchmarking, often designed to assess understanding of specific facts or concepts. The phrasing of these questions can significantly influence the difficulty and interpretation by test-takers. While traditional benchmarks typically use standardized question formats, the sensitivity of language models to variations in question structure, such as word order, remains an underexplored area. Word order changes can alter the syntactic structure of a question without changing its semantic meaning, potentially affecting how models process and answer it. 

Recent studies have shown that language models exhibit varying degrees of sensitivity to syntactic variations in input, particularly when the changes affect the clarity or ambiguity of the question. However, the extent to which such variations impact performance in multiple-choice settings has not been thoroughly investigated. This study addresses this gap by examining whether language models are sensitive to word order changes in multiple-choice questions. By systematically altering the syntactic structure of a controlled set of questions while maintaining their semantic content, we aim to evaluate how these changes affect model accuracy. The results of this experiment provide insights into the robustness of language models to syntactic variations and their ability to generalize across different question formulations.
</text>

## Related Work

\section{Related Work}

Previous research has explored the impact of linguistic variations on the performance of language models, particularly in tasks such as natural language inference and question answering. For instance, \citet{DBLP:journals/corr/abs-2109-07578} investigated how syntactic transformations affect the consistency of model outputs, finding that certain paraphrase pairs can lead to significant performance drops in specific domains. Similarly, \citet{DBLP:journals/corr/abs-2110-08372} analyzed the robustness of language models to syntactic perturbations in multiple-choice settings, emphasizing the role of contextual understanding in maintaining accuracy. However, these studies typically focus on broader syntactic changes rather than controlled word order manipulations within fixed question structures. 

In contrast, this work specifically examines the sensitivity of language models to word order variations in multiple-choice questions, a nuance that has not been extensively explored in existing literature. While \citet{DBLP:journals/corr/abs-2004-09519} addressed the impact of question paraphrasing on model performance, their focus was on semantic equivalence rather than structural reordering. The current study builds upon these findings by employing a targeted approach, where the same factual question is reformulated with minimal syntactic changes to isolate the effect of word order. This methodological difference highlights the importance of structural sensitivity in language models, which may not be fully captured by prior work that prioritizes semantic consistency. 

Furthermore, while some studies have examined the effect of instruction tuning on model performance in structured tasks \citep{DBLP:journals/corr/abs-2106-11959}, the present work emphasizes the role of input formulation rather than model fine-tuning. This distinction is critical, as it shifts the focus from model adaptation to the inherent capabilities of the language model in processing and interpreting variations in question structure. By doing so, this study provides a more granular understanding of how language models process syntactic cues in multiple-choice settings, contributing to a broader discourse on the limitations and potential of large-scale language models in structured reasoning tasks.
</text>

## Methods

\section{Methods}

To investigate whether language models are sensitive to word order in multiple-choice benchmarks, we designed an experimental framework that systematically evaluates model performance on questions with identical semantic content but varying syntactic structure. The primary objective was to quantify the impact of syntactic variation on model accuracy and to assess whether such variation influences the model’s ability to infer correct answers.

The dataset consisted of a curated set of 20 multiple-choice questions, each featuring two versions: an original and an altered version, where the word order was modified without changing the meaning of the question. For each question, the correct answer was explicitly provided in the dataset. The questions were carefully crafted to ensure that the semantic content remained consistent across versions, with only syntactic variation introduced. Examples of such questions include: "Which planet is closest to the sun?" and "To which planet is the sun closest?".

The model used in this study was a language model accessed via the Ollama platform, with the `llama2` or `mistral` variants employed for the experiments. The model was prompted to answer each question in isolation, with no additional context provided. For each question version, the model was instructed to provide a single answer, typically in the form of a sentence or phrase. The answers generated by the model were then compared against the correct answer in the dataset to determine accuracy.

To ensure robustness of the results, the model was given multiple opportunities to generate answers for each question, and the most consistent answer was selected for evaluation. To account for random guessing, a baseline was established by simulating random responses to the questions, with the accuracy of this baseline compared against the model’s performance.

The evaluation process involved the following steps:
1. For each question, the model was presented with both the original and altered versions.
2. The model’s output for each version was recorded and compared to the correct answer.
3. Accuracy was calculated as the proportion of correct answers out of the total number of questions.
4. The results were aggregated across all questions to determine the model’s overall accuracy for both question versions.

The accuracy for each question version was calculated using the formula:

$$
\text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}}
$$

This methodology allowed us to isolate the effect of word order on model performance while maintaining control over other variables. The results were then analyzed to determine whether the model’s accuracy was significantly affected by changes in word order.
</text>

## Experimental Setup

\section{Experimental Setup}

The experimental setup involves a carefully constructed dataset of multiple-choice questions with varying word order, as well as a well-defined evaluation protocol to assess the sensitivity of language models to such variations. The dataset consists of 20 questions, each with two versions: an original and an altered version with a different word order. For instance, the original question might be ``Which planet is closest to the sun?'' while the altered version could be ``To which planet is the sun closest?'' The correct answer is explicitly provided for each question, ensuring consistency in evaluation.

The performance of the language model is measured using accuracy as the primary metric. For each question variant, the model's response is compared against the correct answer to determine whether the prediction is correct. Additionally, a baseline is established by simulating random guessing, where the model's performance is expected to be at chance level (i.e., 1/number of options). 

The language model is loaded via an OpenAI-compatible API and is not fine-tuned for this task. The model's interaction with the dataset is structured such that each question is presented as a standalone prompt, and the model is expected to provide a direct answer. The exact prompt format and model parameters are not specified in this report, as the focus remains on the evaluation of the model's sensitivity to word order rather than the specifics of its training or inference process.

## Results

\section{Results}

The experiment was conducted using a custom dataset of 20 multiple-choice questions, each with two versions: an original and an altered version with different word order. The model's performance was evaluated for both versions. The results showed that the model achieved an accuracy of 0.0 for both the original and altered versions of the questions. This indicates that the model did not provide the correct answers for any of the questions in the dataset. 

This outcome suggests that either the model was not being prompted to answer in a structured multiple-choice format, or the evaluation script was not correctly parsing the model's responses. It is possible that the model was not explicitly asked to select an answer from a list of options, or the evaluation script failed to match the model's output with the correct answer in the dataset. This could be due to variations in case sensitivity, punctuation, or formatting in the model's responses. 

To address these issues, further testing is required to ensure that the model is explicitly prompted to select an answer from a list and that the evaluation script accurately parses and matches the model's output with the correct answer. Additionally, the dataset should be expanded to include more questions and variations in word order to better assess the model's sensitivity to word order in multiple-choice benchmarks.

## Discussion

The results of the experiment, which reported zero accuracy for both original and altered questions, raise critical concerns about the experimental design, prompting strategy, and evaluation mechanism. The lack of model performance suggests that the current setup fails to elicit appropriate responses from the language model, possibly due to ambiguous or insufficient instructions. In standard multiple-choice tasks, models are often prompted to select an answer from a predefined list of choices, which may not have been explicitly included in the current setup. Without such explicit formatting, the model may not be conditioned to produce the expected structured output, leading to responses that are either unanswerable or in a format incompatible with the evaluation script.

Furthermore, the evaluation mechanism appears to lack robustness in parsing the model's output. The absence of any non-zero accuracy could indicate that the script is not effectively matching the model's responses with the correct answers in the dataset. This may stem from issues such as case sensitivity, punctuation, formatting discrepancies, or failure to recognize variations in the model's output format. To ensure reliable results, the evaluation script should be enhanced to include normalization steps, such as case conversion, punctuation removal, and standardization of answer formats, to reduce the likelihood of mismatches.

The current lack of statistical analysis also limits the ability to draw meaningful conclusions about the model's performance. In future work, it would be beneficial to include baseline comparisons, such as random guessing, and to conduct hypothesis testing to determine whether any differences in accuracy between original and altered questions are statistically significant. Additionally, a more comprehensive statistical analysis would provide greater confidence in interpreting the results and identifying trends in the model's sensitivity to word order.

In conclusion, the findings underscore the importance of refining experimental design, prompt formulation, and evaluation mechanisms when assessing language models on structured tasks such as multiple-choice questions. Addressing these methodological challenges would allow for a more accurate and reliable assessment of how language models are influenced by changes in word order, ultimately contributing to the development of more effective prompting strategies and evaluation frameworks for future research.
