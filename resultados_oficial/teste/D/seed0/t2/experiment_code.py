import pandas as pd
from datasets import load_dataset
import os
from openai import OpenAI
import re
import numpy as np
import random
from scipy.stats import ttest_rel

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load the SST-2 dataset with the test split and take the first 200 examples
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Check the columns and print one example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Format the data into a pandas DataFrame
df = pd.DataFrame(dataset)

# Ensure the columns are named 'question' and 'gold_answer' as required
# For SST-2, the 'sentence' column contains the text, and the 'label' column contains the sentiment (0 or 1)
# We will use 'sentence' as the question and 'label' as the gold answer
df.rename(columns={'sentence': 'question', 'label': 'gold_answer'}, inplace=True)

# Display the first few rows of the DataFrame to confirm the formatting
print(df.head())

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key='x')

# Function to evaluate the language model
def evaluate_model(prompt, gold_answers):
    responses = []
    for question, gold_answer in zip(df['question'][:50], gold_answers[:50]):
        messages = [{'role': 'user', 'content': prompt.format(question=question)}]
        r = client.chat.completions.create(
            model=os.environ['AGENTLAB_LLM_MODEL'],
            messages=messages,
            max_tokens=64,
            temperature=0
        )
        answer = r.choices[0].message.content.strip()
        responses.append(answer)
    return responses

# Function to extract the final answer from the response
def extract_answer(answer, gold_answer):
    # Extract the final answer using a regex to find the last number
    match = re.search(r'\b\d\b', answer)
    if match:
        extracted_answer = int(match.group())
    else:
        extracted_answer = -1
    return extracted_answer

# Function to calculate accuracy
def calculate_accuracy(responses, gold_answers):
    correct = 0
    total = len(responses)
    for response, gold_answer in zip(responses, gold_answers):
        extracted_answer = extract_answer(response, gold_answer)
        if extracted_answer == gold_answer:
            correct += 1
    return correct / total

# Evaluate the model with and without negation words
negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'neither', 'nor', 'can\'t', 'won\'t', 'don\'t', 'didn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hadn\'t', 'shouldn\'t', 'wouldn\'t', 'mustn\'t', 'can\'t', 'oughtn\'t'}

# Prompt with negation words
prompt_with_negation = "Classify the sentiment of the following sentence as 0 (negative) or 1 (positive). The sentence is: {question}. Consider the effect of negation words like 'not', 'no', 'never', 'none', etc. Answer:"

# Prompt without negation words
prompt_without_negation = "Classify the sentiment of the following sentence as 0 (negative) or 1 (positive). The sentence is: {question}. Ignore any negation words and focus on the overall sentiment. Answer:"

# Evaluate the model with and without negation words
responses_with_negation = evaluate_model(prompt_with_negation, df['gold_answer'])
responses_without_negation = evaluate_model(prompt_without_negation, df['gold_answer'])

# Calculate accuracy
accuracy_with_negation = calculate_accuracy(responses_with_negation, df['gold_answer'])
accuracy_without_negation = calculate_accuracy(responses_without_negation, df['gold_answer'])

# Print the results
print(f"Accuracy (with negation words): {accuracy_with_negation:.2f}")
print(f"Accuracy (without negation words): {accuracy_without_negation:.2f}")