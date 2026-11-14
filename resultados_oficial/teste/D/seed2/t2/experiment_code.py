import pandas as pd
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from openai import OpenAI
import os
import random
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load the SST-2 dataset and select a test split of at least 200 examples
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Check the columns and print one example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Display the first few rows to confirm the structure
print("First few rows of the dataset:")
print(df.head())

# Initialize the language model
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key='x')

def predict_sentiment(text):
    response = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{'role': 'user', 'content': f'Classify the sentiment of the following text as positive or negative. Output only the label: positive or negative.\nText: {text}'}],
        max_tokens=64,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return 'positive' if 'positive' in answer else 'negative'
    answer = response.choices[0].message.content.strip()
    return 'positive' if 'positive' in answer else 'negative'

# Function to remove negation words from a text
def remove_negation_words(text):
    negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor'}
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in negation_words]
    return ' '.join(filtered_words)

# Function to compute accuracy for a condition
def compute_accuracy(df, condition):
    true_labels = df['label'].apply(lambda x: 'positive' if x == 1 else 'negative')
    predictions = []
    for index, row in df.iterrows():
        text = row['sentence']
        if condition == 'with_negation':
            input_text = text
        elif condition == 'without_negation':
            input_text = remove_negation_words(text)
        else:
            raise ValueError("Invalid condition")
        prediction = predict_sentiment(input_text)
        predictions.append(prediction)
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Compute accuracy for both conditions
accuracy_with_negation = compute_accuracy(df, 'with_negation')
accuracy_without_negation = compute_accuracy(df, 'without_negation')

# Print results
print(f"Accuracy with negation words: {accuracy_with_negation:.4f}")
print(f"Accuracy without negation words: {accuracy_without_negation:.4f}")
print(f"Difference in accuracy: {accuracy_with_negation - accuracy_without_negation:.4f}")