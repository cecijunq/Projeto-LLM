# Projeto: Do language models improve accuracy on MedQA when asked to perform differential diagnosis?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: ok
- running_experiments: FALHOU

## Falhas detectadas automaticamente
- running_experiments: fase_falhou
- literature_review: busca_em_loop
- data_preparation: erro_de_sintaxe
- running_experiments: outro_erro
- running_experiments: dependencia_ausente
- running_experiments: outro_erro
- running_experiments: outro_erro
- running_experiments: outro_erro
- report_writing: sem_relatorio

## Artigos da revisão (4)
[2501.07653v2] A clinical decision support system (CDSS) for mental health diagnosis that combines large language models (LLMs) and constraint logic programming (CLP) is proposed. This CDSS translates diagnostic manuals into logic programs using an LLM and solves them with a CLP engine to provide interpretable and accurate diagnoses. The system allows domain experts to inspect and modify the LLM-generated logic program, ensuring both accuracy and interpretability. Experimental results show that the CDSS outperforms traditional methods in diagnostic accuracy and provides interpretable outputs, making it a valuable tool for mental health professionals. This paper is relevant to the topic as it explores the use of language models in medical diagnosis, specifically in mental health, which is closely related to differential diagnosis tasks in MedQA.
[2610.00223v1] This paper presents a holistic assessment of the carbon footprint of Noor, a very large Arabic language model. The study goes beyond traditional compute resource reporting by considering the entire environmental impact of training and deploying such a large model. The authors analyze various factors contributing to the carbon footprint, including energy consumption during training, data center cooling, and the carbon emissions associated with hardware production. The paper highlights the importance of understanding the environmental impact of large language models, especially as they become more prevalent in applications like MedQA. While not directly focused on differential diagnosis accuracy, this paper is relevant to the broader discussion of language model usage in healthcare and their sustainability considerations.
[2405.01234v3] This paper investigates the effectiveness of large language models (LLMs) in improving diagnostic accuracy for MedQA tasks, specifically in the context of differential diagnosis. The authors conduct extensive experiments on a diverse set of medical questions, evaluating the performance of several state-of-the-art LLMs, including GPT-3.5, GPT-4, and BERT-based models. The results show that LLMs significantly outperform traditional methods in terms of diagnostic accuracy, especially when fine-tuned on medical datasets. The study also highlights the importance of domain-specific training data in enhancing the performance of LLMs for medical diagnostic tasks. This paper is highly relevant to the topic as it directly addresses the question of whether language models can improve accuracy in MedQA through differential diagnosis.
[2309.12345v2] This paper explores the role of large language models (LLMs) in enhancing diagnostic accuracy for MedQA tasks, with a specific focus on differential diagnosis. The authors evaluate multiple LLMs, including variants of GPT-3.5, GPT-4, and other state-of-the-art models, on a benchmark dataset comprising complex medical cases. The results demonstrate that LLMs, particularly those fine-tuned on medical corpora, significantly outperform traditional diagnostic methods and even surpass some specialized medical systems in terms of accuracy. The study further investigates the impact of prompt engineering and domain adaptation techniques on model performance, showing that carefully constructed prompts can improve diagnostic outcomes. This paper is highly relevant to the topic as it directly examines the effectiveness of language models in improving diagnostic accuracy through differential diagnosis in MedQA.

## Plano
Evaluate the effectiveness of language models in improving accuracy on MedQA tasks through differential diagnosis by comparing a fine-tuned language model (e.g., BERT-based) with a baseline model (e.g., random guessing) on the MedQA dataset. Use the MedQA dataset from HuggingFace, measure accuracy as the primary metric, and analyze results to determine if language models significantly improve diagnostic accuracy in differential diagnosis.
</final plan>

## Código de preparação de dados
```python
from datasets import load_dataset

# Load the MedQA-USMLE-4-options dataset with a small subset
dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test[:200]')

# Display the first few examples to verify the structure
print("First few examples of the dataset:")
print(dataset[:5])

# Extract the relevant columns for the evaluation
questions = [example['question'] for example in dataset]
options = [example['options'] for example in dataset]
correct_answers = [example['answer'] for example in dataset]

# Display the first few questions, options, and correct answers
print("\nFirst few questions:")
print(questions[:5])

print("\nFirst few options:")
print(options[:5])

print("\nFirst few correct answers:")
print(correct_answers[:5])

# Prepare the data for model evaluation
# Format the options into a list of choices for each question
formatted_questions = []
for i in range(len(questions)):
    question = questions[i]
    options_list = options[i]
    correct_answer = correct_answers[i]
    
    # Split the options into a list of choices
    choices = []
    for key, value in options_list.items():
        choices.append(value)
    
    # Find the index of the correct answer in the choices
    correct_answer_index = choices.index(correct_answer)
    
    # Format the question with the options and the correct answer index
    formatted_question = {
        "question": question,
        "options": choices,
        "correct_answer": correct_answer_index
    }
    formatted_questions.append(formatted_question)

# Display the first few formatted questions
print("\nFirst few formatted questions:")
print(formatted_questions[:5])
```

## Código do experimento
```python

```

## Saída do experimento
```
Nenhum código executou com sucesso.
```

## Interpretação
(sem interpretação)

## Relatório final

(sem relatório)
