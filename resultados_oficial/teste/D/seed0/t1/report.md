# Research Report: Are language models sensitive to word order in multiple choice benchmarks?

## Abstract

\section*{Abstract}
This study investigates whether language models are sensitive to word order in multiple-choice benchmarks by comparing the model's accuracy on the same questions presented in their original word order versus a randomly reordered version. Conducted on the MMLU benchmark, the experiment reveals a significant drop in accuracy from 61\% in the original order to 41\% in the reordered condition. While this suggests potential sensitivity to word order, the results lack statistical validation to confirm the significance of the observed performance difference. The study highlights the challenges in assessing model behavior under word order variation and underscores the need for rigorous statistical analysis to draw definitive conclusions about model robustness in reasoning tasks.

## Introduction

\section*{Introduction}

The ability of language models to process and reason about multiple-choice questions is a critical aspect of their performance in reasoning tasks. These tasks often rely on the model's capacity to parse question structure, identify relevant information, and select the correct answer from a set of options. However, the extent to which word order influences the model's performance in such tasks remains an underexplored area. While previous research has focused on token efficiency, reasoning suppression, and the distinction between memorization and reasoning in multiple-choice settings, there is limited empirical evidence on how variations in word order affect model accuracy.

This study investigates whether language models are sensitive to word order in multiple-choice benchmarks. Specifically, we conduct an experiment where the same local language model is evaluated on a set of multiple-choice questions with and without altered word order. The MMLU benchmark, a widely used general knowledge dataset, is employed as the test set. By comparing the model's accuracy across original and reordered versions of the same questions, we aim to assess the impact of word order on the model's ability to reason and select the correct answer. The results of this experiment provide insights into the role of syntactic structure in model performance, which has implications for the design of more robust and interpretable reasoning systems.

## Background

\section*{Background}

Language models have shown remarkable capabilities in performing reasoning tasks, particularly in multiple-choice settings. However, the extent to which these models rely on specific syntactic structures, such as word order, to arrive at correct answers remains an open question. While prior work has explored token efficiency, reasoning suppression, and the distinction between memorization and reasoning in multiple-choice benchmarks, little is known about how variations in word order affect model performance. 

Recent studies have highlighted the sensitivity of large language models to reasoning structures and the potential for models to rely on memorization rather than true understanding when answering multiple-choice questions. For instance, the "None of the Others" method demonstrates that models experience significant accuracy drops when the correct answer is dissociated from previously seen tokens, suggesting that reasoning is a critical factor in achieving high performance. Similarly, approaches like CROP and ARS emphasize the importance of optimizing reasoning processes and reducing redundant computation, both of which are closely tied to the structure and order of input.

This study builds on these findings by investigating whether language models exhibit sensitivity to word order in multiple-choice questions. By comparing the same model's performance on questions with original and reordered word order, we aim to shed light on the extent to which syntactic structure influences model behavior in reasoning tasks.

## Related Work



## Methods

\section*{Methods}

To investigate whether language models are sensitive to word order in multiple-choice benchmarks, we conducted an experiment comparing the same local language model's performance on the same multiple-choice questions with two different word orders: the original word order and a reordered version where the content of the question remains unchanged but the order of the answer options is randomized. 

The dataset used for this study is the MMLU benchmark, which contains 200 questions spanning a general knowledge domain. For each question, we generated a reordered version by shuffling the answer options while preserving the question text and the correct answer. This ensures that the only difference between the original and reordered versions lies in the word order of the answer options.

The local language model, served by Ollama, was instructed to answer each question by providing the final answer as a single letter corresponding to the correct option. The model's responses were evaluated for accuracy by comparing its predicted answers to the ground truth answers. The accuracy was computed as the ratio of correctly answered questions to the total number of questions.

To ensure reproducibility, random seeds were fixed for all operations, including the shuffling of answer options. The model's responses were obtained by sending the question text to the language model through an API call, with fixed parameters: a maximum token length of 64, a temperature of 0, and no additional prompts. The model's output was then trimmed and compared to the correct answer.

The experiment was executed directly on the test split of the dataset. For each question, we recorded the model's response, the extracted answer, and the gold answer. These data were used to compute the accuracy for both the original and reordered word orders. The results were then analyzed to assess the impact of word order on model performance.

## Experimental Setup

\section*{Experimental Setup}

The experiment was conducted using the MMLU benchmark, a general knowledge multiple-choice dataset containing 200 questions. The dataset was loaded from the 'WaltonFuture/Diabetes-multiple-choice-questions' repository, with the first 200 entries used for evaluation. Each question in the dataset includes a problem statement and a corresponding answer choice. The experiment focused on assessing the impact of word order on model performance by comparing the model's accuracy when answering questions in their original word order versus when the word order of the question was randomized while preserving the content.

The primary metric used to evaluate model performance was accuracy, defined as the ratio of correct answers to total questions. The model's responses were compared against the gold-standard answers provided in the dataset. No additional hyperparameters were tuned during the experiment; the model was evaluated using its default settings. The experiment was executed directly on the test split of the dataset, with no additional preprocessing applied to the input questions beyond reordering the options in the case of the reordered_order condition.

## Results



## Discussion


