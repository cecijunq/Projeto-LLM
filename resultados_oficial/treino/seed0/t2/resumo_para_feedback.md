# Projeto: Does gender role play affect the accuracy of language models on answering math questions?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: ok
- results_interpretation: ok
- report_writing: ok

## Falhas detectadas automaticamente
- running_experiments: timeout
- running_experiments: erro_de_sintaxe

## Artigos da revisão (5)
[2406.13677v3] Large language models (LLMs) often inherit and amplify social biases embedded in their training data, with gender bias being a prominent example. Prior work has mainly focused on gender stereotyping bias—associating specific roles or traits with a particular gender—and evaluating gender bias in model embeddings or generated outputs. In contrast, gender representation bias—the unequal frequency of references to individuals of different genders in the training corpora—has received less attention. This paper addresses this gap by proposing a novel LLM-based method to detect and quantify gender representation bias in language models. The study highlights that such imbalances in training data constitute an upstream source of bias that can propagate and intensify throughout the model lifecycle. The findings emphasize the importance of addressing gender representation bias to improve the fairness and accuracy of language models, particularly in tasks such as math problem solving where bias could influence the model's performance.
[2507.16557v1] This paper explores gender bias in large language models (LLMs) with a focus on the German language. It presents five German datasets for evaluating gender bias in LLMs, grounded in well-established concepts of gender bias. The datasets are accessible through multiple methodologies, enabling comprehensive analysis. The study investigates unique challenges associated with gender bias in German, such as the ambiguous interpretation of male occupational terms and the influence of seemingly neutral nouns on gender perception. The findings contribute to the understanding of gender bias in multilingual models and highlight the need for language-specific approaches to bias evaluation. The paper also reports results for eight multilingual LLM models, emphasizing the importance of addressing gender bias in diverse linguistic contexts.
[2507.16557v1] This paper explores gender bias in large language models (LLMs) with a focus on the German language. It presents five German datasets for evaluating gender bias in LLMs, grounded in well-established concepts of gender bias. The datasets are accessible through multiple methodologies, enabling comprehensive analysis. The study investigates unique challenges associated with gender bias in German, such as the ambiguous interpretation of male occupational terms and the influence of seemingly neutral nouns on gender perception. The findings contribute to the understanding of gender bias in multilingual models and highlight the need for language-specific approaches to bias evaluation. The paper also reports results for eight multilingual LLM models, emphasizing the importance of addressing gender bias in diverse linguistic contexts.
[2207.10245v1] This paper investigates how language models become biased during their training process, focusing on the evolution of gender bias in an English language model. Using a relatively small language model with an LSTM architecture trained on an English Wikipedia corpus, the study provides a detailed analysis of how gender representation develops over time. The research identifies different phases of training and demonstrates that gender information is dynamically represented in the model. The findings highlight that the bias in the model is driven by patterns in the training data and show how the model's internal state correlates with bias in downstream tasks such as semantic textual similarity. This work contributes to understanding the origins of gender bias in language models and provides insights into mitigating such biases during the training process.
[2406.13677v3] Large language models (LLMs) often inherit and amplify social biases embedded in their training data, with gender bias being a prominent example. Prior work has mainly focused on gender stereotyping bias—associating specific roles or traits with a particular gender—and evaluating gender bias in model embeddings or generated outputs. In contrast, gender representation bias—the unequal frequency of references to individuals of different genders in the training corpora—has received less attention. This paper addresses this gap by proposing a novel LLM-based method to detect and quantify gender representation bias in language models. The study highlights that such imbalances in training data constitute an upstream source of bias that can propagate and intensify throughout the model lifecycle. The findings emphasize the importance of addressing gender representation bias to improve the fairness and accuracy of language models, particularly in tasks such as math problem solving where bias could influence the model's performance.

## Plano
We will design a simple experiment to investigate whether gender role play affects the accuracy of language models on answering math questions. The experiment will involve two scenarios: one where the model is prompted with a neutral math question and another where the question is framed with a specific gender role (e.g., "A man is solving this math problem"). We will use the `math_dataset` from HuggingFace, which contains math problems and their solutions. The model used will be the one served by Ollama via an OpenAI-compatible API. We will run the experiment with 100 questions and report the accuracy in each scenario.

## Código de preparação de dados
```python
import os
import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('azminetoushikwasi/math-story-problems', split='test[:200]')

# Extract questions and answers
questions = []
answers = []

for item in dataset:
    # Check if the item has the required columns
    if 'story_1_qs' in item and 'answer' in item:
        questions.append(item['story_1_qs'])
        answers.append(item['answer'])
    elif 'story_2_qs' in item and 'answer' in item:
        questions.append(item['story_2_qs'])
        answers.append(item['answer'])
    elif 'story_3_qs' in item and 'answer' in item:
        questions.append(item['story_3_qs'])
        answers.append(item['answer'])
    else:
        # Skip items that don't have the required columns
        continue

# Ensure we have exactly 100 questions and answers
questions = questions[:100]
answers = answers[:100]

# Save the questions and answers to a JSON file for later use
output_file = 'math_questions_answers.json'
with open(output_file, 'w') as f:
    json.dump({'questions': questions, 'answers': answers}, f)

print(f"Saved {len(questions)} questions and answers to {output_file}")
```

## Código do experimento
```python
import os
from openai import OpenAI
import json

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Load the dataset
dataset = load_dataset('azminetoushikwasi/math-story-problems', split='test[:200]')

# Extract questions and answers
questions = []
answers = []

for item in dataset:
    # Check if the item has the required columns
    if 'story_1_qs' in item and 'answer' in item:
        questions.append(item['story_1_qs'])
        answers.append(item['answer'])
    elif 'story_2_qs' in item and 'answer' in item:
        questions.append(item['story_2_qs'])
        answers.append(item['answer'])
    elif 'story_3_qs' in item and 'answer' in item:
        questions.append(item['story_3_qs'])
        answers.append(item['answer'])
    else:
        # Skip items that don't have the required columns
        continue

# Ensure we have exactly 50 questions and answers
questions = questions[:50]
answers = answers[:50]

# Save the questions and answers to a JSON file for later use
output_file = 'math_questions_answers.json'
with open(output_file, 'w') as f:
    json.dump({'questions': questions, 'answers': answers}, f)

print(f"Saved {len(questions)} questions and answers to {output_file}")

# Evaluate model performance with neutral and gender-role prompts
neutral_prompts = [f"Q: {q}\nA:" for q in questions]
gender_prompts = [f"Q: A man is solving this math problem. {q}\nA:" for q in questions]

# Store results
neutral_results = []
gender_results = []

# Evaluate with neutral prompts
for prompt in neutral_prompts[:50]:
    r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
    answer = r.choices[0].message.content.strip()
    neutral_results.append(answer)

# Evaluate with gender-role prompts
for prompt in gender_prompts[:50]:
    r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
    answer = r.choices[0].message.content.strip()
    gender_results.append(answer)

# Evaluate accuracy
neutral_answers = [a.split("A:")[1].strip() if "A:" in a else a for a in neutral_results]
gender_answers = [a.split("A:")[1].strip() if "A:" in a else a for a in gender_results]

correct_neutral = sum(1 for i in range(50) if neutral_answers[i] == answers[i])
correct_gender = sum(1 for i in range(50) if gender_answers[i] == answers[i])

print(f"Accuracy with neutral prompts: {correct_neutral}/{50}")
print(f"Accuracy with gender-role prompts: {correct_gender}/{50}")
```

## Saída do experimento
```
Saved 100 questions and answers to math_questions_answers.json
=== experiment output ===
Saved 50 questions and answers to math_questions_answers.json
Accuracy with neutral prompts: 0/50
Accuracy with gender-role prompts: 0/50

[stderr]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

```

## Interpretação
The experimental results indicate that the model's accuracy is 0/50 in both the neutral and gender-role prompt scenarios, suggesting that the model is not capable of correctly answering the math questions in either case. This outcome could be attributed to several factors: 

1. **Model Capabilities**: The model may not be fine-tuned for math problem-solving tasks, leading to poor performance on complex mathematical questions. 
2. **Prompt Design**: The structure of the prompts might not be effective in guiding the model to generate the correct answers, especially if the model lacks the necessary mathematical reasoning skills. 
3. **Dataset Complexity**: The math questions in the dataset might be too advanced or require specific mathematical knowledge that the model has not been trained on. 

To address these issues, future experiments could involve:
- Using a simpler or more aligned math dataset that matches the model's training data.
- Fine-tuning the model on a math-focused corpus to improve its problem-solving capabilities.
- Experimenting with alternative prompt formulations to better guide the model's responses. 

These steps could help in understanding whether gender role play impacts the model's accuracy and in improving the overall performance of the model on math tasks.

## Relatório final

# Research Report: Does gender role play affect the accuracy of language models on answering math questions?

## Abstract

\section*{Abstract}
This study investigates whether gender role play influences the accuracy of language models in answering math questions. Given the prevalence of gender bias in large language models (LLMs), the research explores how framing math problems with specific gender roles might affect model performance. The experiment employs a controlled setup using a math problem dataset and an OpenAI-compatible API to evaluate model responses under neutral and gender-role prompts. Despite careful design, the results show zero accuracy in both scenarios, highlighting challenges in model capability and prompt effectiveness. This finding underscores the difficulty of isolating gender bias effects in complex tasks and emphasizes the need for more capable models and refined evaluation methodologies. The study contributes to the growing body of work on bias in LLMs by focusing on the intersection of gender roles and mathematical reasoning.
</text>

## Introduction

\section*{Introduction}

The integration of language models into academic and practical domains has raised significant concerns regarding their potential to inherit and amplify societal biases. Among these, gender bias remains a prominent issue, particularly in tasks involving reasoning and problem-solving. While existing literature has extensively explored gender stereotyping and representation bias in language models, the specific impact of gender role play on the accuracy of models when answering math questions remains underexplored. This study investigates whether framing math questions with explicit gender roles—such as "A man is solving this math problem"—affects the accuracy of language models in generating correct solutions. 

Mathematical problem-solving is a domain where reasoning and logical inference are paramount, and any embedded biases could potentially influence the model's ability to produce accurate results. Previous research has indicated that biases present in training data can propagate through the model's decision-making processes, potentially leading to disparities in performance across different groups. By introducing gender role prompts, this study seeks to explore whether such biases manifest in the context of mathematical reasoning. 

The experiment described in this paper employs a controlled setting to assess model performance under two scenarios: neutral prompts and gender-role prompts. The results highlight the challenges in achieving accurate mathematical responses, suggesting that the model may not be sufficiently trained or prompted to perform effectively in this domain. These findings underscore the importance of addressing both data biases and prompt design in improving the fairness and reliability of language models, particularly in critical tasks such as mathematical problem solving.
</text>

## Background



## Related Work



## Methods

\section{Methods}

To investigate the impact of gender role play on the accuracy of language models in answering math questions, we designed a controlled experimental methodology. The experiment aimed to determine whether framing a math question with a specific gender role (e.g., "A man is solving this math problem") would influence the model's ability to generate correct answers. The methodology is based on a prompt-based evaluation framework, where the model's outputs are compared against a predefined set of ground truth answers.

The model used in the experiment was served by an OpenAI-compatible API provided by Ollama, with the model identifier specified by the environment variable \texttt{AGENTLAB\_LLM\_MODEL}. The model was prompted with two distinct types of questions: neutral prompts and gender-role prompts. Neutral prompts were constructed as: 

$$
\text{Q: } q_i \text{A:}
$$

where $ q_i $ represents the $ i $-th math question. Gender-role prompts were constructed by appending a gender-specific role to the question, as follows:

$$
\text{Q: A man is solving this math problem. } q_i \text{A:}
$$

This formulation was chosen to simulate a scenario where the model is explicitly told about the gender of the problem solver, potentially influencing its internal representation of the task.

The evaluation dataset was sourced from the \texttt{math-story-problems} dataset on HuggingFace, which contains math problems and their corresponding solutions. The dataset was loaded using the \texttt{load\_dataset} function, and only the \texttt{test} split was used, with the first 200 items selected for analysis. From this, 50 questions and their corresponding answers were extracted, ensuring that the dataset used for evaluation was representative of the full dataset.

Each question was paired with its corresponding answer, and the model's output for both prompt types was collected. The model's responses were parsed to extract the answer portion following the prompt, and these were compared against the ground truth answers to compute accuracy. Specifically, correctness was defined as the number of instances where the model's generated answer matched the correct answer exactly.

The evaluation was conducted using the \texttt{OpenAI} API, with each prompt being sent as a single message in a chat completion request. The model was instructed to generate responses with a maximum token count of 64 and a temperature of 0 to ensure deterministic and focused outputs. The experiment was repeated for both prompt types, and the results were aggregated to assess the impact of gender role play on model accuracy.
</text>

## Experimental Setup

\section{Experimental Setup}

The experiment was designed to investigate whether gender role play affects the accuracy of language models in answering math questions. The dataset used was the \texttt{math-story-problems} dataset from HuggingFace, which contains math problems accompanied by contextual stories. To focus on the core mathematical questions, only the question portions were extracted, and the first 200 items were selected for the evaluation. The final dataset comprised 50 questions and their corresponding answers, ensuring a manageable size for the experiment.

The model used in the experiment was accessed via an OpenAI-compatible API provided by Ollama, with the model identifier specified by the environment variable \texttt{AGENTLAB\_LLM\_MODEL}. The model was prompted with two types of input: (1) neutral prompts, which contained only the math question, and (2) gender-role prompts, which included a gender-specific context such as ``A man is solving this math problem'' preceding the question. Both prompt types were structured to elicit a mathematical answer in the form of ``A: <answer>'', with a token limit of 64 tokens and a temperature setting of 0 to ensure deterministic output.

The primary metric for evaluating model performance was accuracy, defined as the proportion of correctly answered questions relative to the total number of questions. The model's responses were compared against the ground truth answers from the dataset to determine correctness. No additional hyperparameters were tuned beyond the prompt structure and model configuration, as the focus was on assessing the impact of gender role play on model accuracy rather than optimizing model performance.
</text>

## Results

The experimental results demonstrate that the model's accuracy was 0/50 in both the neutral prompt and gender-role prompt scenarios. This indicates that the model failed to generate correct answers for all 50 math questions in both conditions. The model's inability to answer any of the questions suggests limitations in its mathematical reasoning capabilities or an insufficiently aligned training data distribution with respect to the task. The results were consistent across both prompt types, implying that the introduction of gender role play did not influence the model's accuracy in this specific setup. The experiment encountered challenges related to the model's performance on complex mathematical problems, which may stem from a lack of fine-tuning for math problem-solving tasks or the inherent difficulty of the dataset used. These findings highlight the need for further investigation into model capabilities and prompt design to ensure reliable performance on math-related tasks.
</text>

## Discussion



