# Research Report: Are sentiment classifiers sensitive to negation words on short movie reviews (SST-2)?

## Abstract

\section*{Abstract}
This study investigates the sensitivity of sentiment classifiers to negation words in short movie reviews from the SST-2 dataset. Understanding how negation affects sentiment classification is crucial for improving the accuracy of natural language processing systems, particularly in short-text scenarios where linguistic cues are limited. The challenge lies in the subtle and context-dependent nature of negation, which can invert the sentiment polarity of a sentence. We conducted a controlled experiment by evaluating a local model on the SST-2 dataset with and without negation words, aiming to quantify the impact of negation on classification accuracy. However, the experiment yielded no valid results due to a technical error in the evaluation setup, highlighting the need for rigorous validation and careful implementation when assessing sentiment classifiers. Despite the limitations, this work underscores the importance of addressing negation in sentiment analysis and sets the foundation for future research in this area.

## Introduction

\section{Introduction}

Sentiment analysis plays a crucial role in understanding user opinions in natural language text, particularly in domains such as movie reviews where sentiment polarity is a key indicator of viewer satisfaction. Short movie reviews, such as those in the Stanford Sentiment Treebank (SST-2) dataset, present unique challenges due to their brevity and the frequent presence of negation words, which can invert the sentiment expressed in a sentence. Negation words, such as "not" or "never," are known to significantly alter the sentiment polarity of a review, yet their impact on sentiment classifiers remains an area of active research.

This study investigates whether sentiment classifiers are sensitive to negation words in short movie reviews from the SST-2 dataset. While previous work has explored the influence of linguistic modifiers on sentiment composition and the challenges of negation handling in low-resource and morphologically rich languages, there is a need for a more direct empirical examination of how negation affects sentiment classification in short texts. The focus on the SST-2 dataset allows for a controlled evaluation of sentiment models under varying conditions of negation presence, providing insights into their robustness and sensitivity to such linguistic phenomena.

The experimental design involves evaluating a sentiment classifier on the same set of short movie reviews with and without negation words, using a standardized evaluation framework. The results of this experiment will contribute to a deeper understanding of how sentiment classifiers process negation and inform the development of more accurate and robust models for sentiment analysis in short texts.

## Background

\section*{Background}

Sentiment analysis is a fundamental task in natural language processing (NLP), with applications spanning from product reviews to social media monitoring. The Stanford Sentiment Treebank (SST-2) is a widely used benchmark dataset for binary sentiment classification, comprising short movie reviews annotated with positive and negative labels. Despite its popularity, the sensitivity of sentiment classifiers to negation words—such as "not," "never," and "no"—remains an area of active research. Negation words can significantly alter the sentiment polarity of a sentence, often leading to misclassification if not properly handled by models.

Previous studies have shown that the presence of negation words can affect sentiment classification accuracy, particularly in short texts where context is limited. For instance, \cite{1712.01794v1} investigated the impact of negators, modals, and degree adverbs on sentiment composition, finding that the influence of these modifiers varies across different linguistic contexts. Similarly, \cite{1906.07610v2} proposed a multi-task learning approach to explicitly model negation in sentiment analysis, achieving improved performance on short texts such as movie reviews. These findings underscore the importance of understanding how sentiment classifiers process negation, especially in the context of short, concise reviews like those in the SST-2 dataset.

The current study builds on these insights by designing an experiment to evaluate the sensitivity of sentiment classifiers to negation words in the SST-2 dataset. The experiment involves comparing the performance of a model with and without negation words in the input, thereby assessing how the presence of such linguistic elements affects sentiment classification accuracy.

## Related Work

\section*{Related Work}

The sensitivity of sentiment classifiers to negation words has been extensively studied in the context of sentiment analysis, particularly in short texts such as movie reviews. Early work, such as \cite{1712.01794v1}, investigated the impact of negators, modals, and degree adverbs on sentiment composition, revealing that the effect of these modifiers varies significantly across different linguistic contexts. This study emphasized the importance of understanding how such elements influence sentiment, which is crucial for improving the performance of sentiment classifiers in the presence of negation. However, this work primarily focused on annotated phrases rather than full texts, limiting its applicability to short reviews.

In contrast, \cite{2107.11597v1} addressed the challenges of negation handling in low-resource and morphologically rich languages, such as colloquial Arabic. The authors proposed a rule-based approach that leverages linguistic knowledge and sentiment lexicons to improve sentiment classification accuracy. Although this work is tailored to a specific language and setting, its emphasis on the necessity of explicit negation handling is relevant to our study, as it highlights the broader challenge of modeling negation in sentiment analysis. Unlike our experiment, which focuses on the sensitivity of models to negation words, this paper's approach is more rule-based and less data-driven, suggesting that the effectiveness of negation handling may depend on the model's architecture and training data.

A more recent approach, introduced in \cite{1906.07610v2}, proposed a multi-task learning framework that explicitly incorporates negation information into sentiment analysis. This method outperformed implicit negation learning by disentangling the effect of negation on the final polarity of a text. This approach is particularly relevant to our study, as it demonstrates that modeling negation as a distinct task can enhance sentiment classification performance in short texts. However, unlike our experimental setup, which evaluates the sensitivity of a single model to negation words, this work focuses on the integration of negation handling into a broader learning framework, suggesting that the sensitivity to negation may be more context-dependent than previously assumed.

Our experiment builds on these studies by directly testing the impact of negation words on sentiment classification accuracy in a controlled setting. Unlike the rule-based and multi-task learning approaches, which aim to improve overall accuracy through explicit modeling of negation, our experiment evaluates the model's response to the presence or absence of negation words. This distinction is critical, as it allows us to isolate the effect of negation on model performance and provides insights into the model's behavior in the presence of linguistic ambiguity. Ultimately, these studies together highlight the significance of negation in sentiment analysis and underscore the need for robust and interpretable methods to handle its effects.

## Methods

\section{Methods}

The experiment was designed to assess the sensitivity of sentiment classifiers to negation words in short movie reviews from the SST-2 dataset. The methodology follows a controlled comparison approach, where the same model is evaluated on the same dataset under two distinct conditions: with negation words and without negation words. The primary goal is to quantify how the presence of negation affects sentiment classification accuracy.

The SST-2 dataset, a widely used benchmark for sentiment analysis, consists of 200 short movie reviews, each labeled as either positive (label 1) or negative (label -1). For this experiment, the test split of the dataset was used, consisting of 200 examples. The dataset was loaded and preprocessed to extract the sentence and label for each review. The dataset was then split into a training set and a test set, each containing 100 examples, to ensure that the model is evaluated on a representative subset of the data.

The model used for evaluation is a local instance of an LLM served by Ollama, which was configured to operate with a fixed temperature of 0 and a maximum token limit of 64 to ensure deterministic and concise outputs. The model was instructed to classify the sentiment of a given review, either with or without negation words, by generating a response that explicitly states the sentiment label. The prompt for each condition was carefully crafted to ensure that the model is asked to classify the sentiment of the review text, while the presence or absence of negation words is systematically varied.

For each condition, the model's output was parsed to extract the predicted label, which was then compared to the ground truth label to compute classification accuracy. The accuracy was calculated using the standard accuracy metric, defined as:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}.
$$

The difference in accuracy between the two conditions was then analyzed to determine whether the model's performance was significantly affected by the presence of negation words. If the accuracy dropped significantly when negation words were removed, it would indicate that the model is sensitive to negation in its sentiment classification. Conversely, if the accuracy remained unchanged, it would suggest that the model is not sensitive to negation words.

To ensure reproducibility, the random seed was set to 42 before loading the dataset and initializing the model. All outputs were recorded, and the first three examples for each condition were printed to provide a qualitative assessment of the model's behavior. The results were then analyzed to draw conclusions about the sensitivity of the sentiment classifier to negation words in short movie reviews.

## Experimental Setup

\section{Experimental Setup}

The experiment was conducted using the Stanford Sentiment Treebank (SST-2) dataset, a widely used benchmark for sentiment analysis tasks. Specifically, the test split of the dataset was utilized, containing 200 short movie reviews annotated with binary sentiment labels (positive or negative). Each review was represented by a sentence and a corresponding label, where label $-1$ indicates negative sentiment and label $1$ indicates positive sentiment. 

The primary metric for evaluating model performance was accuracy, defined as the proportion of correctly classified reviews. To assess the sensitivity of the sentiment classifier to negation words, the model was tested under two conditions: one where negation words were present in the input text and another where these words were removed. The comparison of accuracy between the two conditions aimed to determine whether the presence of negation words significantly influenced the model's predictions.

The model used for evaluation was a large language model hosted by Ollama, with the specific model identifier provided through the environment variable. No hyperparameters were explicitly tuned for this experiment, as the focus was on the effect of negation words rather than model optimization. The inference process was performed using a prompt-based approach, where the model was instructed to classify the sentiment of a given review text. 

The implementation relied on standard Python libraries and the Hugging Face datasets library for data loading and preprocessing. All random seeds were set for reproducibility, and the data was split into training and test sets to ensure consistency in the evaluation procedure. The experimental setup was designed to isolate the impact of negation words on sentiment classification, providing a clear framework for analyzing the model's behavior in the presence of such linguistic features.

## Results

\section{Results}

The experiment was conducted to assess the sensitivity of a sentiment classifier to negation words in short movie reviews from the SST-2 dataset. The evaluation was performed using a local model served by Ollama, with the same set of 200 test examples processed under two conditions: with negation words and without negation words. The metric used for evaluation was accuracy.

The results showed that the model achieved an accuracy of 0.00 in both conditions, indicating that the model failed to correctly classify the sentiment of the reviews in all cases. The output for the first three examples in both conditions is presented below:

\textbf{With negation words:}
Example 1:
Model Response: 0
Gold Answer: -1

Example 2:
Model Response: 0
Gold Answer: -1

Example 3:
Model Response: 0
Gold Answer: -1

\textbf{Without negation words:}
Example 1:
Model Response: 0
Gold Answer: -1

Example 2:
Model Response: 0
Gold Answer: -1

Example 3:
Model Response: 0
Gold Answer: -1

The model's inability to produce accurate classifications suggests a potential issue with the experimental setup or the model's configuration. The low accuracy observed in both conditions precludes any meaningful comparison of the model's sensitivity to negation words. The lack of valid results limits the ability to draw conclusions about the impact of negation on sentiment classification performance in this context.

## Discussion

The results of the experiment indicate a critical failure in the evaluation setup, as the model produced an accuracy of 0.00 in both conditions, regardless of the presence or absence of negation words. This suggests that either the model's responses were not properly parsed, or there was an issue with the data processing pipeline. The uniformity in the model's output across all examples raises concerns about the reliability of the evaluation method. It is possible that the model's output was not correctly mapped to the expected label format, leading to incorrect accuracy calculations. Additionally, the use of a non-standard prompt format may have contributed to the model's inability to provide meaningful classifications. Given the lack of valid results, it is not possible to draw conclusions about the sensitivity of sentiment classifiers to negation words in this experiment. Future work should focus on refining the evaluation setup, ensuring proper parsing of model outputs, and validating the model's ability to distinguish between different sentiment labels. Furthermore, the use of a more robust sentiment classification model or a different prompting strategy may be necessary to obtain meaningful insights into the impact of negation on sentiment classification performance. The failure of this experiment highlights the importance of careful implementation and validation in experimental design, particularly when dealing with complex linguistic phenomena such as negation in sentiment analysis. Addressing these technical challenges will be essential for producing reliable and informative results in future studies on the sensitivity of sentiment classifiers to negation words in short movie reviews.
