import pandas as pd
from datasets import load_dataset
import os
from openai import OpenAI

# Load the GSM8K dataset from HuggingFace
dataset = load_dataset('openai/gsm8k', 'main', split='test[:100]')

# Convert to a pandas DataFrame for easier handling
df = pd.DataFrame(dataset)

# Display the first few rows to verify the data
print(df.head())

# Save the dataset to a CSV file for later use
df.to_csv('gsm8k_subset.csv', index=False)

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Prepare the dataset for evaluation
problems = df['question'].tolist()
solutions = df['answer'].tolist()

# Define prompting strategies
baseline_prompt = "Solve the following math problem:\n{question}\nAnswer: "
cot_prompt = "Reason step by step and then solve the following math problem:\n{question}\nAnswer: "

# Evaluate the model with both prompting strategies
correct_answers = 0
total_answers = 0

print("Evaluating Baseline Prompting...")
for i in range(min(len(problems), 50)):  # Limit to 50 queries
    question = problems[i]
    solution = solutions[i]
    
    # Format the prompt
    formatted_prompt = baseline_prompt.format(question=question)
    
    # Query the model
    r = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{'role': 'user', 'content': formatted_prompt}],
        max_tokens=64,
        temperature=0
    )
    answer = r.choices[0].message.content.strip()
    
    # Check for correctness
    if answer == solution:
        correct_answers += 1
    total_answers += 1
for i in range(min(len(problems), 50)):  # Limit to 50 queries
    question = problems[i]
    solution = solutions[i]
    
    # Format the prompt
    formatted_prompt = baseline_prompt.format(question=question)
    
    # Query the model
    r = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{'role': 'user', 'content': formatted_prompt}],
        max_tokens=64,
        temperature=0
    )
    answer = r.choices[0].message.content.strip()
    
    # Check for correctness
    if answer == solution:
        correct_answers += 1
    total_answers += 1

print(f"Baseline Accuracy: {correct_answers}/{total_answers} ({correct_answers/total_answers*100:.2f}%)")

correct_answers = 0
total_answers = 0

print("Evaluating Chain-of-Thought (CoT) Prompting...")
for i in range(min(len(problems), 50)):  # Limit to 50 queries
    question = problems[i]
    solution = solutions[i]
    
    # Format the prompt
    formatted_prompt = cot_prompt.format(question=question)
    
    # Query the model
    r = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{'role': 'user', 'content': formatted_prompt}],
        max_tokens=64,
        temperature=0
    )
    answer = r.choices[0].message.content.strip()
    
    # Check for correctness
    if answer == solution:
        correct_answers += 1
    total_answers += 1

print(f"CoT Accuracy: {correct_answers}/{total_answers} ({correct_answers/total_answers*100:.2f}%)")