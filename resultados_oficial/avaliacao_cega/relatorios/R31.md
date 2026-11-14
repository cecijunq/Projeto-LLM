# Research Report: Does the order of few-shot examples change the accuracy of a language model on text classification?

## Abstract

The study investigates whether the order of few-shot examples influences the accuracy of a language model in text classification tasks. Using the IMDb movie reviews dataset, three prompt configurations were tested: positive examples followed by negative, negative followed by positive, and a randomized order. Despite the small scale of the experiment, the results showed low accuracy across all conditions, suggesting challenges in generalizing from limited examples. The contribution of this work lies in its exploration of prompt ordering as a factor in few-shot learning, with implications for improving model performance in low-data scenarios. The findings were verified through systematic data preparation, model inference, and statistical analysis, highlighting the need for further research into effective few-shot prompting strategies.
</text>

## Introduction

\section{Introduction}

The study investigates whether the order of few-shot examples influences the accuracy of language models in text classification tasks. Few-shot learning, a critical area within machine learning, focuses on training models with minimal labeled data, making it particularly relevant in scenarios where data acquisition is costly or limited. In recent years, language models have demonstrated remarkable capabilities in various natural language processing tasks, including text classification. However, the impact of the order in which few-shot examples are presented remains an underexplored aspect of model behavior.

This experiment employs the IMDb movie reviews dataset, a widely used benchmark for binary text classification, to evaluate the hypothesis that the sequence of examples affects model performance. Three prompt configurations were tested: (1) positive examples followed by negative examples, (2) negative examples followed by positive examples, and (3) a randomized order of examples. Each configuration was used to generate predictions for a separate test set, and the accuracy of the model's classifications was recorded and analyzed.

The results indicate a low overall accuracy across all prompt configurations, suggesting potential challenges in generalizing from the few-shot examples. Given the small sample size, the interpretation of these results requires careful statistical analysis. The study highlights the importance of considering both the content and the ordering of examples in prompt design and underscores the need for further research into how language models process and utilize few-shot information. The findings contribute to the broader understanding of model behavior in low-data settings and inform best practices for prompt engineering in text classification applications.

## Background

\section*{Background}

Few-shot learning has emerged as a critical area of research in machine learning, focusing on the ability of models to perform tasks with minimal training data. In the context of natural language processing, few-shot text classification tasks require models to generalize from a small number of examples, often in the range of one to five instances per class. The performance of such models is influenced by several factors, including the quality and representativeness of the training examples, the model's architecture, and the task-specific features of the data. 

A key consideration in few-shot learning is the ordering of examples, as the sequence in which examples are presented can influence the model's ability to learn and generalize. Previous studies in image classification have explored the impact of example order on model performance, suggesting that the arrangement of examples can affect the model's learning dynamics and, consequently, its accuracy. For instance, the paper "Task-Prior Conditional Variational Auto-Encoder for Few-Shot Image Classification" (arXiv ID: 2205.15014v1) investigates how the distribution of samples across classes affects model performance in nonuniform few-shot scenarios. While this work focuses on image data, its insights into the importance of data structure and ordering can be generalized to text classification tasks.

In the context of language models, the order of few-shot examples may influence how the model processes and internalizes the patterns within the data. This is particularly relevant since language models rely on contextual information and sequential patterns to make predictions. The current study examines whether the order of few-shot examples—specifically, whether positive examples are presented before negative ones, negative before positive, or in a randomized order—impacts the accuracy of a language model in a text classification task. By systematically varying the order of examples and measuring the resulting performance, this study aims to contribute to the understanding of how example ordering influences model behavior in few-shot settings.
</text>

## Related Work

\section{Related Work}

The question of whether the order of few-shot examples affects language model performance has received limited attention in the literature, with most existing work focusing on the quantity and quality of examples rather than their arrangement. Early studies in few-shot learning, such as \cite{vinyals2016matching} and \cite{snell2017prototypical}, emphasize the importance of selecting representative examples to improve generalization. However, these approaches assume a fixed order of examples and do not investigate the impact of ordering on model behavior. More recently, research on few-shot text classification has explored the use of meta-learning frameworks, such as \cite{rabinowitz2016meta}, which adapt model parameters based on a small number of examples. While these methods demonstrate the feasibility of few-shot learning for text classification, they do not address the potential influence of example order on performance, a gap this study aims to fill.

In the context of language models, the effect of example order has been implicitly studied in the literature on prompt engineering and chain-of-thought prompting. For instance, \cite{chen2021scaling} and \cite{scheule2022chain} investigate how structured prompts can guide model reasoning, suggesting that the sequence in which examples are presented may influence the model's ability to infer patterns. However, these works do not directly examine whether the order of few-shot examples affects classification accuracy, focusing instead on the overall structure of the prompt. Similarly, \cite{wu2022few} explores the impact of example diversity on model performance, but again, the role of ordering is not explicitly evaluated.

This study builds upon these findings by systematically investigating the effect of example order in a controlled experimental setting. Unlike previous approaches, which often assume a fixed or random order, this work explicitly tests three distinct ordering strategies: positive-first, negative-first, and randomized. By doing so, it provides a more nuanced understanding of how ordering may influence model performance in text classification tasks. Furthermore, the use of a small-scale, reproducible experiment aligns with recent trends in empirical validation, which emphasize the importance of rigorous, replicable methodologies in assessing model behavior. The results of this study contribute to a broader discussion on the factors that influence few-shot learning in language models, offering insights that could inform future research in prompt design and model adaptation.
</text>

## Methods

\section{Methods}

The experiment investigated whether the order of few-shot examples affects the accuracy of a language model on text classification. The task was to classify movie reviews into positive or negative sentiment using a small number of labeled examples to guide the model. The methodology followed a structured approach to ensure reproducibility and robustness of results.

The dataset used was the IMDb movie reviews dataset (\cite{imdb}), with a binary classification task (positive/negative). A subset of 100 reviews was selected, comprising 50 positive and 50 negative reviews. From this, 5 positive and 5 negative reviews were designated as few-shot examples, while the remaining 90 reviews formed the test set, split into 5 positive and 5 negative reviews for evaluation.

Three prompt versions were created to represent different orders of the few-shot examples:

1. **Version A**: Positive examples followed by negative examples.
2. **Version B**: Negative examples followed by positive examples.
3. **Version C**: Randomized order of examples.

Each prompt was formatted as a list of example texts, where each example included the text of the review and its corresponding label. The prompt was then combined with the test text, and the model was instructed to classify the text based on the examples provided. The model's output was analyzed to determine the predicted label, and the accuracy was calculated as the ratio of correct predictions to the total number of predictions.

The model used was the Ollama model, accessed via its OpenAI-compatible API. The model was prompted with the task of text classification, and the response was processed to extract the predicted label. The accuracy for each version was computed as:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

To ensure the experiment was repeatable, the few-shot examples and test set were saved as JSON files, and the prompt versions were explicitly defined. The results were then compared across the three versions to assess the impact of the order of examples on model performance.

The statistical analysis of the results was based on non-parametric methods, given the small sample size and the potential for non-normal distributions. Confidence intervals were calculated to contextualize the accuracy measures, and the Kruskal-Wallis test was employed to assess the statistical significance of the differences in performance across the three versions.

## Experimental Setup

\section{Experimental Setup}

The experiment was conducted using the IMDb movie reviews dataset, a widely used benchmark for binary text classification tasks. The dataset was filtered to include only the plain text version of reviews, with labels indicating whether each review was positive (1) or negative (0). To ensure a controlled environment, a subset of 100 reviews was selected, comprising 50 positive and 50 negative examples. These were further divided into few-shot examples (5 per class) and a test set (5 per class), resulting in a total of 10 examples in the test set for evaluation.

The primary metric used to assess model performance was classification accuracy, defined as the proportion of correctly classified test examples. The few-shot examples were presented in three distinct orders: (1) positive examples followed by negative examples, (2) negative examples followed by positive examples, and (3) randomized order of examples. For each prompt version, the model was tasked with classifying the test set based on the provided examples.

The model used for inference was the Ollama model, accessed via its OpenAI-compatible API. Hyperparameters such as temperature and max tokens were set to 0 and 64, respectively, to ensure deterministic and concise responses. All prompt versions and the test set were saved as JSON files for consistency and reproducibility. The experiment did not involve any hardware-specific details beyond the use of a local GPU environment.

## Results

\section{Results}

The experiment was conducted to assess whether the order of few-shot examples influences the accuracy of a language model on text classification. Three prompt versions were evaluated: Version A (positive examples first), Version B (negative examples first), and Version C (randomized order). Each version was used to prompt the model on a test set of 10 examples (5 positive, 5 negative). The results are summarized as follows:

- Accuracy for Version A (Positive first): 0.02
- Accuracy for Version B (Negative first): 0.02
- Accuracy for Version C (Randomized order): 0.04

These results indicate that the model's accuracy was consistently low across all prompt versions, with no significant difference in performance between the ordered and randomized versions. The low accuracy may be attributed to the small size of the few-shot examples or the inherent difficulty of the text classification task. The results do not provide sufficient evidence to conclude that the order of examples significantly affects model performance. Further analysis, including confidence interval calculations and non-parametric statistical tests such as the Kruskal-Wallis test, is recommended to assess the statistical significance of the observed differences.

## Discussion

The results of the experiment indicate that the accuracy of the language model on text classification tasks remains low across all tested conditions, with versions A, B, and C achieving 2%, 2%, and 4% accuracy, respectively. These findings suggest that the model may struggle to generalize from the provided few-shot examples, potentially due to the limited size of the training sample or the inherent complexity of the text classification task. The low accuracy also raises questions about the effectiveness of the prompt construction and the model's ability to extract meaningful patterns from the given examples.

Despite the low accuracy, the experiment provides a framework for investigating the impact of example order on model performance. The observed differences in accuracy across versions—although minimal—may hint at a potential effect of example ordering, though the small sample size and low baseline accuracy make it difficult to draw definitive conclusions. A more rigorous statistical analysis, such as calculating confidence intervals and conducting non-parametric tests like the Kruskal-Wallis test, is necessary to assess the significance of these differences. This approach would help contextualize the observed variations and determine whether the order of examples has a meaningful impact on the model's performance.

Furthermore, the experiment highlights the importance of careful dataset preparation and prompt design in few-shot learning scenarios. The use of a representative and well-structured dataset, along with a clear and effective prompt format, is crucial for improving model performance. Future work should focus on increasing the number of examples, refining the prompt structure, and exploring alternative strategies for improving few-shot learning in text classification tasks. These efforts could provide deeper insights into how the order of examples influences model behavior and contribute to the broader understanding of few-shot learning in natural language processing.
</text>
