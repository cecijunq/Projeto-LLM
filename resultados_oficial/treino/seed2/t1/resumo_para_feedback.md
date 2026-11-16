# Projeto: Do language models improve accuracy on MedQA when asked to perform differential diagnosis?

## Etapas
- literature_review: ok
- plan_formulation: ok
- data_preparation: FALHOU

## Falhas detectadas automaticamente
- data_preparation: fase_falhou
- literature_review: formato_invalido
- data_preparation: erro_de_sintaxe
- data_preparation: erro_de_sintaxe
- report_writing: sem_relatorio

## Artigos da revisão (5)
[2501.07653v2] This paper presents a clinical decision support system (CDSS) for mental health diagnosis that combines large language models (LLMs) with constraint logic programming (CLP). The CDSS translates diagnostic manuals into a logic program using an LLM and solves it with a CLP engine to provide interpretable and accurate diagnoses. The system allows domain experts to inspect and modify the LLM-generated logic program, ensuring accuracy and interpretability. The paper demonstrates the effectiveness of this approach in mental health diagnosis, highlighting the potential of LLMs in clinical settings.
[2503.21676v2] This paper investigates the learning dynamics of language models on a synthetic factual recall task, uncovering three key findings: First, language models learn in three phases, exhibiting a performance plateau before acquiring precise factual knowledge. Second, the training data distribution significantly impacts learning dynamics, as imbalanced distributions lead to shorter plateaus. Finally, hallucinations emerge simultaneously with knowledge, and integrating new knowledge into the model through fine-tuning is challenging, as it quickly corrupts existing knowledge. These findings provide insights into how language models acquire and retain factual knowledge, which is critical for understanding their behavior in tasks such as differential diagnosis.
[2501.07653v2] This paper presents a clinical decision support system (CDSS) for mental health diagnosis that combines large language models (LLMs) with constraint logic programming (CLP). The CDSS translates diagnostic manuals into a logic program using an LLM and solves it with a CLP engine to provide interpretable and accurate diagnoses. The system allows domain experts to inspect and modify the LLM-generated logic program, ensuring accuracy and interpretability. The paper demonstrates the effectiveness of this approach in mental health diagnosis, highlighting the potential of LLMs in clinical settings.
[2501.07653v2] This paper presents a clinical decision support system (CDSS) for mental health diagnosis that combines large language models (LLMs) with constraint logic programming (CLP). The CDSS translates diagnostic manuals into a logic program using an LLM and solves it with a CLP engine to provide interpretable and accurate diagnoses. The system allows domain experts to inspect and modify the LLM-generated logic program, ensuring accuracy and interpretability. The paper demonstrates the effectiveness of this approach in mental health diagnosis, highlighting the potential of LLMs in clinical settings.
[2501.07653v2] This paper presents a clinical decision support system (CDSS) for mental health diagnosis that combines large language models (LLMs) with constraint logic programming (CLP). The CDSS translates diagnostic manuals into a logic program using an LLM and solves it with a CLP engine to provide interpretable and accurate diagnoses. The system allows domain experts to inspect and modify the LLM-generated logic program, ensuring accuracy and interpretability. The paper demonstrates the effectiveness of this approach in mental health diagnosis, highlighting the potential of LLMs in clinical settings.

## Plano
1. **Objective**: Investigate whether language models improve accuracy on MedQA when performing differential diagnosis.
2. **Dataset**: Use the MedQA dataset to formulate differential diagnosis questions.
3. **Baseline**: Use a rule-based system or expert knowledge-based approach as the baseline for comparison.
4. **LLM Setup**: Use Ollama to serve the language model (e.g., Llama 3) locally without downloading additional models.
5. **Steps**:
   - Load the MedQA dataset.
   - Extract differential diagnosis questions from the dataset.
   - Query the LLM for answers using Ollama's API.
   - Compare the LLM's responses with the correct answers.
   - Calculate accuracy metrics for both the LLM and the baseline.
6. **Evaluation**: Analyze the results to determine if the LLM improves accuracy on MedQA for differential diagnosis.
</final plan>

## Código de preparação de dados
```python

```

## Código do experimento
```python

```

## Saída do experimento
```

```

## Interpretação
(sem interpretação)

## Relatório final

(sem relatório)
