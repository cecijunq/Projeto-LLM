from datasets import load_dataset
import pandas as pd
import time
import os
from openai import OpenAI

# Load the dataset
dataset = load_dataset('AlgorithmicResearchGroup/math_reasoning_automated_problem_solving_with_code_track_3', 'test', split='train[:200]')

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Display the first few rows to verify the data
print("First few rows of the dataset:")
print(df.head())

# Save the dataset to a CSV file for further use (optional)
df.to_csv('math_questions.csv', index=False)

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Experimental conditions
conditions = [
    {"condition": "Baseline", "prompt": "Solve the following math problem: [problem]"},
    {"condition": "Gender Role Play (Male)", "prompt": "You are a male student who is good at math. Solve the following math problem: [problem]"},
    {"condition": "Gender Role Play (Female)", "prompt": "You are a female student who is good at math. Solve the following math problem: [problem]"},
    {"condition": "Control", "prompt": "You are a student who is good at math. Solve the following math problem: [problem]"}
]

# Evaluation metrics
metrics = [
    "Accuracy (correct answer vs. generated answer)",
    "Response time",
    "Confidence level (if available)"
]

# Iterate over each problem in the dataset
for index, row in df.iterrows():
    # Check if 'problem' column exists
    if 'problem' not in df.columns:
        print("Error: 'problem' column not found in the dataset.")
        continue
    
    problem = row['problem']
    correct_answer = row['solution']
    print(f"\nProblem: {problem}")
    print(f"Correct Answer: {correct_answer}")
    
    # Run experiments for each condition
    for condition in conditions:
        prompt = condition["prompt"].replace("[problem]", problem)
        print(f"\nStarting experiment: {condition['condition']}")
        start_time = time.time()
        r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{"role": "user", "content": prompt}], max_tokens=256, temperature=0)
        answer = r.choices[0].message.content
        end_time = time.time()
        response_time = end_time - start_time
        
        # Check accuracy
        is_correct = answer.strip() == correct_answer.strip()
        accuracy = "Correct" if is_correct else "Incorrect"
        
        # Print results
        print(f"Experiment: {condition['condition']}")
        print(f"Answer: {answer}")
        print(f"Accuracy: {accuracy}")
        print(f"Response Time: {response_time:.4f} seconds")