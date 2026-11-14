import os
from openai import OpenAI
import pandas as pd
import random

# Load the dataset
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Define the three example orders
def random_order(df, n_examples_per_class=5):
    df = df.sample(frac=1).reset_index(drop=True)
    return df.groupby('label').head(n_examples_per_class).reset_index(drop=True)

def class_sorted_order(df, n_examples_per_class=5):
    df = df.sort_values('label').reset_index(drop=True)
    return df.groupby('label').head(n_examples_per_class).reset_index(drop=True)

def baseline_order(df):
    return df.head(0)  # No examples provided

# Generate the three example orders
random_examples = random_order(df)
class_sorted_examples = class_sorted_order(df)
baseline_examples = baseline_order(df)

# Function to format the prompt
def format_prompt(examples, task_description):
    prompt = task_description + "\n\n"
    for _, row in examples.iterrows():
        prompt += f"Text: {row['sentence']}\nLabel: {row['label']}\n\n"
    return prompt

# Task description
task_description = "Classify the following text into positive or negative sentiment."

# Format the prompts
random_prompt = format_prompt(random_examples, task_description)
class_sorted_prompt = format_prompt(class_sorted_examples, task_description)
baseline_prompt = format_prompt(baseline_examples, task_description)

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Function to get model predictions
def get_predictions(prompt, n_samples=50):
    predictions = []
    for _ in range(n_samples):
        r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
        answer = r.choices[0].message.content
        # Extract the predicted label
        if 'Label:' in answer:
            predicted_label = answer.split('Label:')[-1].strip()
            if 'positive' in predicted_label.lower():
                predicted_label = 1
            elif 'negative' in predicted_label.lower():
                predicted_label = 0
            else:
                predicted_label = 0  # Default to negative
        else:
            predicted_label = 0  # Default to negative
        predictions.append(predicted_label)
    return predictions

# Get predictions for each prompt
random_predictions = get_predictions(random_prompt, n_samples=50)
class_sorted_predictions = get_predictions(class_sorted_prompt, n_samples=50)
baseline_predictions = get_predictions(baseline_prompt, n_samples=50)

# Compute accuracy
def compute_accuracy(predictions, true_labels):
    return sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)

# Extract true labels
true_labels = [row['label'] for _, row in df.iterrows()]

# Compute accuracy for each condition
random_accuracy = compute_accuracy(random_predictions, true_labels)
class_sorted_accuracy = compute_accuracy(class_sorted_predictions, true_labels)
baseline_accuracy = compute_accuracy(baseline_predictions, true_labels)

# Print results
print("Random Order Accuracy:", random_accuracy)
print("Class-Sorted Order Accuracy:", class_sorted_accuracy)
print("Baseline Accuracy:", baseline_accuracy)