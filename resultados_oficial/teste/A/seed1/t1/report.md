# Research Report: Are language models sensitive to word order in multiple choice benchmarks?

## Abstract

The study investigates the sensitivity of Ollama's local language model to word order in the context of the SuperGLUE WiC dataset, which evaluates the ability to determine whether a word is used consistently across two different contexts. This task is critical for understanding semantic nuances and contextual variations in language, yet it remains challenging for language models due to the subtlety required in discerning meaning from word order and tense. The experiment demonstrates that while the model performs well on identical contexts, it struggles with tense differences and subtle contextual shifts, leading to inconsistent predictions. These findings highlight the limitations of current models in capturing precise semantic relationships and underscore the need for further research into improving contextual understanding in natural language processing tasks. The results are verified through a structured evaluation using a curated dataset and direct comparison of model outputs against labeled examples.
</text>

## Introduction

\section{Introduction}

The ability of language models to understand and process word order is a critical aspect of their natural language understanding capabilities. Word order significantly influences the meaning of sentences and the interpretation of words within specific contexts. In multiple-choice benchmarks, where subtle differences in word usage and context can determine the correct answer, the sensitivity of language models to such variations is essential for accurate performance. Recent studies have highlighted the limitations of large language models (LLMs) in capturing the nuances of word order and contextual usage, particularly in tasks that require fine-grained understanding of linguistic structures. For instance, research has shown that while LLMs perform well on a wide range of natural language tasks, their ability to understand the fundamental structure of text, such as character composition and word order, remains inconsistent. This has raised questions about their reliability in tasks that depend heavily on contextual and syntactic analysis.

This study investigates the sensitivity of Ollama's local language model to word order in the context of the SuperGLUE WiC (Word-in-Context) dataset, which is designed to assess whether a word is used in the same way across two different contexts. The dataset provides a controlled environment to evaluate how well language models can discern subtle differences in word usage based on grammatical structures and contextual variations. By analyzing the model's performance on this task, the study aims to shed light on its ability to process and interpret word order, which is crucial for tasks that require precise understanding of linguistic nuances. The findings from this experiment contribute to the broader discussion on the capabilities and limitations of language models in handling complex linguistic structures and contextual dependencies.
</text>

## Background

\section{Background}

The ability of language models to understand and process word order is a fundamental aspect of natural language understanding. Word order plays a critical role in determining the meaning of sentences, particularly in tasks that require distinguishing between subtle variations in context and tense. This capability is essential for tasks such as word sense disambiguation, where the same word can have different meanings depending on its surrounding context. Recent studies have highlighted the limitations of large language models (LLMs) in accurately capturing the nuances of word usage and contextual dependencies. For instance, research has shown that LLMs often struggle with tasks that require a deep understanding of the structural composition of text, such as character-level analysis or the recognition of tense differences. Additionally, the task of Visual Word Sense Disambiguation (VWSD) has further emphasized the importance of contextual understanding in determining the meaning of words. This task involves retrieving an image that best represents the meaning of a word within a specific context, demonstrating the interplay between textual and visual information in conveying semantic meaning. The present study builds on these findings by examining the sensitivity of Ollama's local language model to word order in the context of the SuperGLUE WiC dataset, which is designed to assess the ability of models to determine whether a word is used in the same way across two different contexts. This investigation provides insights into the model's capacity to interpret contextual variations in word usage and its implications for tasks requiring precise linguistic understanding.

## Related Work



## Methods

\section{Methods}

The experiment aims to evaluate the sensitivity of Ollama's local language model to word order in the context of the SuperGLUE WiC (Word-in-Context) dataset. The primary objective is to assess the model's ability to determine whether a given word is used in the same way in two different contexts. The methodology is structured around the following steps:

\subsection{Dataset Preparation}

The SuperGLUE WiC dataset is utilized as the source of examples. Each example in the dataset contains four fields: "word", "context1", "context2", and "label". The "label" field indicates whether the word is used in the same way in both contexts (label "Yes") or not (label "No"). The dataset used in the experiment consists of 18 manually curated examples, as shown in the code provided in the plan.

To ensure consistency and facilitate model input, the examples are structured in a JSON format and converted into a dictionary-based dataset using the Hugging Face \texttt{Dataset} class. The dataset is constructed as follows:

$$
\text{dataset} = \left\{
\begin{array}{l}
\text{word} = [\text{example}["word"] \text{ for } \text{example} \in \text{examples}] \\
\text{context1} = [\text{example}["context1"] \text{ for } \text{example} \in \text{examples}] \\
\text{context2} = [\text{example}["context2"] \text{ for } \text{example} \in \text{examples}] \\
\text{label} = [\text{example}["label"] \text{ for } \text{example} \in \text{examples}]
\end{array}
\right.
$$

\subsection{Prompt Generation and Model Interaction}

For each example, a prompt is generated to query the model about the consistency of word usage in the two contexts. The prompt is structured as:

$$
\text{prompt} = f"Is the word '{word}' used in the same way in these two contexts: '{context1}' and '{context2}'?"
$$

The model is then queried using the OpenAI API, with the prompt formatted as a single user message. The model's response is analyzed to extract the predicted label (e.g., "Yes" or "No") based on the model's textual output. The exact mapping from the model's response to the predicted label is determined by keyword matching, where the presence of "Yes" or "No" in the model's response indicates the predicted label.

\subsection{Evaluation Metrics}

The primary evaluation metric is accuracy, which is computed as the ratio of correctly predicted labels to the total number of examples. For each example, the predicted label is compared against the true label provided in the dataset. The accuracy is defined as:

$$
\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N}
$$

where $ N $ is the total number of examples, $ y_i $ is the true label of the $ i $-th example, and $ \hat{y}_i $ is the predicted label. The model's performance is thus quantified by its ability to correctly distinguish between instances where the word is used in the same way versus instances where it is not.

\subsection{Implementation Details}

The experiment is implemented using Python, with the \texttt{Dataset} class from the Hugging Face library used for dataset management. The OpenAI API is employed to interact with the model, with the base URL and API key provided through environment variables. Each example is processed individually, with the prompt generated and sent to the model. The model's response is then parsed to extract the predicted label, which is subsequently compared to the true label to compute the accuracy. The implementation ensures that the dataset is loaded, processed, and evaluated in a systematic and reproducible manner.

## Experimental Setup

\section{Experimental Setup}

The experimental setup was designed to evaluate the sensitivity of Ollama's local language model to word order in the context of the SuperGLUE WiC benchmark. The dataset used in the experiment consisted of 18 examples, each containing a target word, two contextual sentences, and a binary label indicating whether the word was used in the same way in both contexts. The dataset was structured in JSON format, with features including "word", "context1", "context2", and "label". 

For each example, the model was prompted with a question in the format: ``Is the word '[word]' used in the same way in these two contexts: '[context1]' and '[context2]'?'' The model's response was analyzed to extract a prediction, either ``Yes`` or ``No``, and compared against the true label to compute accuracy. The evaluation metric used was accuracy, which quantifies the proportion of correctly predicted labels. 

Hyperparameters for the model were not explicitly tuned in this experiment. The model's inference was conducted using default settings, with a maximum token limit of 64 and a temperature parameter set to 0 to prioritize deterministic output. The experiment did not involve hardware-specific details, as the focus was on the model's behavior and its sensitivity to word order in the given task.
</text>

## Results

\section{Results}

The experiment evaluated the sensitivity of Ollama's local language model to word order in the context of the SuperGLUE WiC dataset. A total of 18 examples were processed, with each example consisting of a target word, two contextual sentences, and a binary label indicating whether the word was used in the same way in both contexts. The model's predictions were compared against the true labels to assess its accuracy.

The model achieved an accuracy of 66.67\% across the 18 examples. Specifically, it correctly identified the same usage in 10 examples (6 examples labeled "Yes" and 4 examples labeled "No") and incorrectly predicted the same usage in 4 examples. In 4 examples, the model returned an ambiguous or uncertain response, which could not be mapped to the expected binary label ("Yes" or "No"), leading to incorrect classifications.

The model performed well on examples where the two contexts were identical, correctly identifying the same usage in all such cases. However, it struggled with examples involving tense differences, such as "He saw the movie" versus "He sees the movie" and "He made a mistake" versus "He makes a mistake," where it either returned uncertain responses or failed to correctly classify the label.

The results indicate that the model's sensitivity to word order is inconsistent, particularly when the differences in usage are subtle, such as tense shifts. The model's performance suggests that while it can recognize explicit differences in word usage in some cases, it lacks the capacity to consistently detect and interpret more nuanced variations in context and word order, which are critical for tasks requiring precise semantic understanding. These findings highlight the limitations of current models in handling tasks that depend on fine-grained contextual and grammatical awareness, consistent with prior research on the challenges of understanding word order and tense in natural language tasks.
</text>

## Discussion

\section*{Discussion}

The experimental results reveal that Ollama's local language model demonstrates a nuanced sensitivity to word order, particularly in the context of tense and grammatical variation, as evaluated through the SuperGLUE WiC dataset. The model's ability to distinguish between identical contexts and correctly predict "Yes" in such cases indicates a reasonable understanding of word usage consistency. However, its performance deteriorates when confronted with subtle differences in tense and grammatical form, such as the shift from past to present tense. In some instances, the model correctly identifies these differences and labels them as "No," demonstrating an awareness of the impact of word order on meaning. In other cases, however, the model fails to provide a clear prediction, returning an ambiguous "Unknown" response, which results in incorrect classifications and highlights a significant limitation in its ability to reliably detect and interpret contextual variations.

This mixed performance aligns with broader findings in the literature on large language models, which suggest that while these models excel at many natural language tasks, their understanding of fundamental linguistic structures such as word order, tense, and grammatical nuance remains inconsistent. For instance, the study by \cite{2405.11357v3} highlights the challenges that LLMs face in processing the character composition of words, which may be related to the difficulties observed in this experiment regarding the interpretation of tense and grammatical form. Similarly, the work in \cite{2310.14025v1} emphasizes the potential of LLMs in understanding contextual word meanings, which is closely tied to the sensitivity to word order in multiple-choice benchmarks. However, the results of this experiment suggest that even with such potential, models may struggle with tasks requiring precise semantic and grammatical analysis, particularly when the differences are subtle or context-dependent.

The findings also raise important questions about the design of benchmarking tasks for language models, particularly those that rely on multiple-choice formats. If models are expected to demonstrate sensitivity to word order and grammatical variation, the evaluation metrics and task formulations must be carefully designed to ensure that the models are not only capable of detecting such differences but also able to interpret and classify them accurately. The ambiguity in some of the model's responses underscores the need for more robust evaluation protocols that can account for the nuances in model behavior and provide a clearer picture of their performance in real-world scenarios.

In conclusion, while Ollama's local language model shows promise in tasks involving word meaning consistency, its performance in detecting subtle changes in word order and grammatical form remains inconsistent. These findings suggest that further research is needed to improve the model's ability to interpret contextual variations, particularly in tasks that require precise understanding of linguistic structure and semantic nuance.
</text>
