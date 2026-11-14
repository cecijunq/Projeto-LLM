import os
from openai import OpenAI
import random
import numpy as np
import pandas as pd
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# Load synthetic-domain-text-classification dataset
dataset = load_dataset('argilla/synthetic-domain-text-classification', split='train[:200]')

# Ensure the dataset has the required columns
if 'text' not in dataset.column_names or 'label' not in dataset.column_names:
    raise ValueError("The dataset does not contain the required columns 'text' and 'label'.")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

# Separate positive and negative examples
positive_examples = df[df['label'] == 1]
negative_examples = df[df['label'] == 0]

# Select 2 examples per class
selected_positive = positive_examples.sample(n=2)
selected_negative = negative_examples.sample(n=2)

# Combine selected examples
few_shot_examples = pd.concat([selected_positive, selected_negative]).sample(frac=1).reset_index(drop=True)

# Generate permutations of few-shot examples
num_permutations = 10
permutations = [few_shot_examples.sample(frac=1).reset_index(drop=True) for _ in range(num_permutations)]

# Initialize OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Define the few-shot prompt format
def create_prompt(permutation):
    prompt = ""
    for index, row in permutation.iterrows():
        text = row['text']
        label = row['label']
        prompt += f"Text: {text}\nLabel: {label}\n\n"
    return prompt

# Define the query format
def format_query(text):
    return f"Classify the following text as positive or negative: {text}"

# Evaluate accuracy across permutations
results = []
for _ in range(5):  # Repeat experiment 5 times
    for permutation in permutations:
        # Create the prompt
        prompt = create_prompt(permutation)
        
        # Format the query for the model
        query = "Classify the following text as positive or negative: [text]"
        
        # Generate the response
        response = client.chat.completions.create(
            model=os.environ['AGENTLAB_LLM_MODEL'],
            messages=[{"role": "user", "content": prompt + query}],
            max_tokens=64,
            temperature=0
        )
        
        # Extract the answer
        answer = response.choices[0].message.content.strip()
        
        # Check if the answer is "positive" or "negative"
        if answer.lower() == "positive":
            predicted_label = 1
        elif answer.lower() == "negative":
            predicted_label = 0
        else:
            predicted_label = None  # Skip if answer is not valid
        
        # Check if the answer is correct
        if predicted_label is not None:
            correct_label = permutation.iloc[0]['label']  # Assuming the first example is the test example
            if predicted_label == correct_label:
                results.append(1)
            else:
                results.append(0)
        else:
            results.append(0)  # Skip if answer is not valid

# Calculate accuracy
accuracy = sum(results) / len(results)
print(f"Accuracy across permutations: {accuracy:.2f}")