# Evaluate the impact of few-shot example order on classification accuracy using the Iris dataset

from sklearn.datasets import load_iris
import random
import numpy as np
from openai import OpenAI
import os

# Load the Iris dataset (a built-in sklearn dataset)
data = load_iris()
# Select 20 examples (10 per class) for few-shot learning
few_shot_data = data.data[:20]
few_shot_labels = data.target[:20]

# Create multiple permutations of the few-shot examples (e.g., 5 permutations)
num_permutations = 5
permutations = []
for _ in range(num_permutations):
    # Shuffle the examples
    shuffled_data = list(zip(few_shot_data, few_shot_labels))
    random.shuffle(shuffled_data)
    # Split into data and labels
    permuted_data = [item[0] for item in shuffled_data]
    permuted_labels = [item[1] for item in shuffled_data]
    permutations.append((permuted_data, permuted_labels))

# Define the test set (remaining 100 examples)
test_data = data.data[20:]
test_labels = data.target[20:]

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Evaluate each permutation
for i, (permuted_data, permuted_labels) in enumerate(permutations):
    # Prepare the prompt with few-shot examples
    prompt = ""
    for d, l in zip(permuted_data, permuted_labels):
        prompt += f"Example: {d} -> Label: {l}\n"
    
    # Add the test examples to the prompt
    prompt += "Test examples:\n"
    for d in test_data:
        prompt += f"Example: {d} -> Label: \n"
    
    # Use the Ollama language model to classify the test examples
    r = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=256,
        temperature=0
    )
    
    # Extract the answers
    answers = r.choices[0].message.content.split('\n')
    
    # Extract the labels from the answers
    predicted_labels = []
    for ans in answers:
        if ans.startswith("Label: "):
            predicted_labels.append(int(ans.split("Label: ")[1]))
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(test_labels, predicted_labels) if true == pred)
    accuracy = correct / len(test_labels)
    
    # Print the results
    print(f"Permutation {i+1} Accuracy: {accuracy:.2f}")