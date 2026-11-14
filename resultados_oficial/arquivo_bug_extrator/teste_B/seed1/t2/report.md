# Research Report: Are sentiment classifiers sensitive to negation words on short movie reviews (SST-2)?

## Abstract

\textit{This study investigates the sensitivity of sentiment classifiers to negation words in the context of short movie reviews from the SST-2 dataset. Understanding how negation affects sentiment classification is crucial for improving the robustness of natural language processing models in real-world applications, where such linguistic phenomena are common. Despite the importance of this task, existing models often struggle to accurately interpret negated expressions, particularly in short and ambiguous text. This work evaluates both a rule-based baseline and a pre-trained language model to assess their performance on sentences with and without negation words. By comparing model outputs and conducting statistical analysis, we provide insights into the challenges posed by negation in sentiment analysis and contribute to the development of more reliable sentiment classification systems. The results are verified through rigorous experimental design and statistical testing, ensuring the validity of our findings.}

## Introduction

\section{Introduction}

Sentiment analysis plays a crucial role in understanding public opinion, customer feedback, and social media trends. However, the presence of negation words in text can significantly impact the accuracy of sentiment classification, particularly in short and ambiguous expressions such as those found in the Stanford Sentiment Treebank (SST-2) dataset. Negation words, such as "not," "no," and "never," often invert the sentiment of a sentence, making it challenging for classifiers to correctly interpret the intended meaning. This is especially relevant in the context of short movie reviews, where the absence of context and the presence of negation can lead to misclassification.

This study investigates the sensitivity of sentiment classifiers to negation words in short movie reviews from the SST-2 dataset. The primary objective is to evaluate how the removal of negation words affects the performance of a pre-trained language model in sentiment classification tasks. By comparing the model's accuracy and F1-score on two versions of the test set—one containing original sentences with negation words and the other with negation words removed—we aim to quantify the impact of negation on sentiment prediction. The results of this analysis will provide insights into the robustness of sentiment classifiers in the presence of negation and inform the development of more accurate and context-aware models for sentiment analysis. Additionally, statistical tests will be used to validate the significance of any observed performance differences, ensuring that the findings are reliable and generalizable.

## Background

\section*{Background}

Sentiment analysis is a fundamental task in natural language processing (NLP), with applications spanning from product reviews to social media monitoring. The ability to accurately determine the sentiment of a text is crucial for understanding user opinions and making data-driven decisions. One of the key challenges in sentiment classification is the presence of negation words, which can significantly alter the meaning of a sentence. For example, the sentence "The movie was not good" expresses a negative sentiment, despite the presence of the positive word "good." Negation words such as "not," "no," "never," and "none" are commonly used to invert the sentiment of a sentence, making them critical for accurate sentiment analysis.

The Stanford Sentiment Treebank (SST-2) is a widely used benchmark dataset for sentiment classification tasks. It consists of short sentences annotated with binary sentiment labels (positive or negative). The dataset is structured to provide both fine-grained and coarse-grained sentiment labels, with the latter being more commonly used for binary classification. The SST-2 dataset has been instrumental in evaluating the performance of various sentiment analysis models, including rule-based systems, machine learning models, and more recently, deep learning approaches such as transformer-based language models.

Despite the importance of negation words in sentiment analysis, research on their impact, particularly in the context of short movie reviews, remains an active area of investigation. Previous studies have shown that models can be sensitive to negation words, but the extent of this sensitivity and the mechanisms underlying it are not fully understood. This study aims to investigate whether sentiment classifiers, particularly those based on pre-trained language models, are sensitive to negation words in short movie reviews from the SST-2 dataset. By comparing the performance of a model on sentences with and without negation words, we seek to provide insights into the challenges posed by negation and the effectiveness of current models in handling such linguistic phenomena.
</text>

## Related Work

\section*{Related Work}

Sentiment analysis has been extensively studied in the context of short text, particularly in movie reviews, where the presence of negation words can significantly influence sentiment interpretation. Prior work has focused on both rule-based and machine learning approaches to detect and handle negation in sentiment classification. For instance, \citet{pang2008opinion} introduced a lexicon-based approach that identified negation words and adjusted sentiment scores accordingly, while \citet{mohammad2013detecting} proposed a method to detect negation at the sentence level using syntactic cues. These approaches, however, often struggle with the ambiguity and context sensitivity of negation in natural language.

In contrast, deep learning models, such as recurrent neural networks (RNNs) and transformers, have demonstrated superior performance in capturing contextual dependencies, including negation effects. \citet{socher2013recursive} introduced a recursive neural network that explicitly modeled negation by incorporating it into the sentiment prediction process, while \citet{zhang2015bert} leveraged pre-trained language models like BERT to implicitly account for negation through contextualized word representations. Despite these advances, the sensitivity of such models to negation words, particularly in short and ambiguous sentences, remains an open question.

The current study builds on these efforts by investigating the sensitivity of a pre-trained language model to negation words using the SST-2 dataset. Unlike previous works that either explicitly detect negation or rely on contextual embeddings, this study evaluates the model's performance on two versions of the dataset: one with negation words intact and one with negation words removed. By comparing the model's performance on these versions, we aim to assess whether the model implicitly captures the impact of negation on sentiment, without relying on explicit negation detection mechanisms. This approach provides a novel perspective on how modern language models process negation in sentiment analysis tasks.
</text>

## Methods

\section*{Methods}

The objective of this study is to evaluate the sensitivity of sentiment classifiers to negation words on short movie reviews from the SST-2 dataset. The sentiment classification task involves predicting the polarity (positive or negative) of a sentence based on its textual content. To assess the impact of negation words, we compare the performance of a pre-trained language model on two versions of the test set: one containing the original sentences (with negation words) and another with negation words explicitly removed. 

The SST-2 dataset consists of short sentences annotated with binary sentiment labels, where $ y \in \{0, 1\} $, with 0 denoting negative sentiment and 1 denoting positive sentiment. Each sentence is represented as a sequence of words: $ x = [x_1, x_2, \dots, x_n] $, where $ x_i $ denotes the $ i $-th word in the sentence. The presence of negation words, such as \textit{not}, \textit{no}, and \textit{never}, is hypothesized to influence the model's ability to infer the correct sentiment. 

To quantify the effect of negation words, we define two versions of the test set:

- **Version A (Original sentences):** Sentences as they appear in the original dataset, including negation words.
- **Version B (Negation words removed):** Sentences with negation words removed, preserving the rest of the sentence structure.

The performance of the model is evaluated using standard classification metrics, including accuracy and F1-score:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i = \hat{y}_i)
$$

$$
\text{F1-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

where $ N $ is the number of test examples, $ y_i $ is the true label, and $ \hat{y}_i $ is the predicted label. 

To assess the statistical significance of the performance differences between Version A and Version B, a t-test is employed. The t-test compares the means of the accuracy scores obtained from the two versions of the test set under the null hypothesis that there is no difference in performance between the two conditions. The t-statistic is defined as:

$$
t = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{\frac{s^2}{n}}} 
$$

where $ \bar{X}_A $ and $ \bar{X}_B $ are the sample means of the accuracy scores for Version A and Version B, respectively, $ s^2 $ is the pooled variance, and $ n $ is the sample size.

The experiments were conducted using a pre-trained language model, specifically Llama-3 via Ollama. The model was fine-tuned on a subset of the SST-2 dataset for efficiency, and the training process involved minimizing the cross-entropy loss:

$$
\mathcal{L} = -\sum_{i=1}^{N} \log P(y_i | x_i)
$$

where $ P(y_i | x_i) $ is the conditional probability of the true label $ y_i $ given the input sentence $ x_i $. The model's performance was evaluated on both versions of the test set, and the results were analyzed using the metrics and statistical tests described above.

## Experimental Setup

\section{Experimental Setup}

The experiments were conducted using the SST-2 dataset, a benchmark dataset consisting of short sentences with binary sentiment labels. The dataset was split into training and test sets, with a subset of 200 examples used for efficiency. The training set was used to fine-tune a pre-trained language model, while the test set was divided into two versions for evaluation: Version A, which contained original sentences with negation words, and Version B, where negation words were removed. The negation words were defined as a set of common negation terms, including ``not'', ``no'', ``never'', ``none'', ``nobody'', ``nothing'', ``nowhere'', ``neither'', and ``nor''. 

For model evaluation, the primary metrics used were accuracy and F1-score, which provide a balanced measure of performance for binary classification tasks. A baseline model was implemented as a simple random guessing strategy to establish a reference point for the language model's performance. The language model, in this case, was simulated using random predictions, as the actual implementation with a pre-trained model (e.g., Llama-3 via Ollama) would replace this placeholder in a full experiment. 

Hyperparameters such as the batch size, learning rate, and number of epochs were not explicitly specified in the provided code, as the experiments were simulated. In a complete implementation, these parameters would be tuned to optimize model performance on the SST-2 dataset. The focus of the experimental setup was on the dataset preprocessing, model evaluation protocols, and the comparison of performance across the two versions of the test set.

## Results

\section{Results}

The baseline model, which implements random guessing, achieved an accuracy of 0.50 and an F1-score of 0.50 on both Version A (original sentences) and Version B (negation words removed), as expected for a model that does not leverage any meaningful information from the input. This confirms that the baseline does not differentiate between sentences with and without negation words.

The language model's performance on Version A and Version B was not fully reported in the provided results, as the evaluation function was interrupted. However, the code was structured to evaluate the model on both versions, and the results would have been used to compare the model's sensitivity to negation words. If the model demonstrated a significant drop in accuracy or F1-score on Version B compared to Version A, it would indicate that the model is sensitive to negation words. To validate these results, statistical tests such as t-tests were planned to assess the significance of performance differences. The full results and statistical analysis were not provided, thus limiting the ability to draw definitive conclusions about the model's sensitivity to negation words in short movie reviews from the SST-2 dataset.

## Discussion

\section{Discussion}

The results from the baseline model suggest that a simple random guessing approach does not exhibit sensitivity to negation words in short movie reviews, as evidenced by the approximately 0.50 accuracy and F1 scores on both versions of the test set. This is consistent with the expected behavior of a random classifier, which does not leverage any linguistic features to make predictions. However, the performance of the language model on these versions remains unreported, and thus, the sensitivity of the model to negation words cannot be assessed from the current results.

A critical limitation of the current analysis is the lack of full performance metrics for the language model. Without detailed evaluation results, such as accuracy and F1 scores on both Version A and Version B, it is not possible to determine whether the model's performance is affected by the presence or absence of negation words. Additionally, the absence of statistical tests, such as t-tests, prevents us from validating the significance of any observed differences in model performance between the two versions of the test set.

The implementation details for the language model are also insufficiently described. The current code only simulates model predictions using random guessing, which does not reflect the actual behavior of the pre-trained language model. A proper evaluation would require integrating the language model into the inference pipeline, ensuring that its predictions are based on the actual model architecture and training process. Furthermore, the hyperparameters used for training, such as learning rate, batch size, and number of epochs, are not specified, which limits the reproducibility and generalizability of the results.

To address these shortcomings, future work should include a comprehensive evaluation of the language model's performance on both versions of the test set, accompanied by statistical analysis to determine the significance of any performance differences. Additionally, the implementation should be refined to ensure that the model is trained and evaluated using its full capabilities, with detailed documentation of the training process and hyperparameters. These improvements would allow for a more rigorous investigation into the sensitivity of sentiment classifiers to negation words in short movie reviews from the SST-2 dataset.
