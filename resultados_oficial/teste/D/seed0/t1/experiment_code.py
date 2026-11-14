import pandas as pd
from datasets import load_dataset
import random
import numpy as np
import os
from openai import OpenAI

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['TORCH_HOME'] = '/tmp/torch_home'

# Load the dataset
dataset = load_dataset('WaltonFuture/Diabetes-multiple-choice-questions', split='train[:200]')

# Check the columns and print an example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Ensure the dataset has the required columns: question and answer
if 'problem' in dataset.column_names and 'answer' in dataset.column_names:
    # Prepare the data for the experiment
    df = pd.DataFrame(dataset)
    print("Data preparation completed.")
else:
    print("The dataset does not have the required columns 'question' and 'answer'.")
    exit()

# Function to generate reordered questions
def reorder_question(problem):
    # Split the problem into lines
    lines = problem.split('\n')
    # Extract the question and options
    question_part = lines[0]
    options = lines[1:]
    # Reorder the options
    random.shuffle(options)
    # Reconstruct the problem with reordered options
    reordered_problem = question_part + '\n' + '\n'.join(options)
    return reordered_problem

# Reorder the questions
df['reordered_problem'] = df['problem'].apply(reorder_question)

# Function to evaluate the model
def evaluate_model(df, order_type):
    total = 0
    correct = 0
    examples = []

    for _, row in df.iterrows():
        # Skip rows where 'problem' or 'answer' is NaN
        if pd.isna(row['problem']) or pd.isna(row['answer']):
            continue

        problem = row['reordered_problem'] if order_type == 'reordered_order' else row['problem']
        answer = row['answer']

        # Format the prompt
        prompt = problem

        # Call the language model
        client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key='x')
        r = client.chat.completions.create(
            model=os.environ['AGENTLAB_LLM_MODEL'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=64,
            temperature=0
        )
        answer_text = r.choices[0].message.content.strip()

        # Extract the answer
        if answer_text == answer:
            extracted_answer = answer
        else:
            extracted_answer = 'Unknown'

        # Store the example
        examples.append({
            'model_response': answer_text,
            'extracted_answer': extracted_answer,
            'gold_answer': answer
        })

        total += 1
        if extracted_answer == answer:
            correct += 1

    return correct, total, examples

# Function to print examples
def print_examples(examples, condition):
    print(f"\nExamples for {condition}:")
    for i, example in enumerate(examples):
        print(f"Example {i+1}:")
        print("Model Response:", example['model_response'])
        print("Extracted Answer:", example['extracted_answer'])
        print("Gold Answer:", example['gold_answer'])
        print()

# Evaluate the model on both conditions
original_correct, original_total, original_examples = evaluate_model(df, 'original_order')
reordered_correct, reordered_total, reordered_examples = evaluate_model(df, 'reordered_order')

# Print the results
print(f"Accuracy (original_order): {original_correct}/{original_total} = {original_correct/original_total:.2f}")
print(f"Accuracy (reordered_order): {reordered_correct}/{reordered_total} = {reordered_correct/reordered_total:.2f}")

# Print examples
print_examples(original_examples, 'original_order')
print_examples(reordered_examples, 'reordered_order')