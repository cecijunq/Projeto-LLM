import pandas as pd
import random
from datasets import load_dataset

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

# Save the prompts to files
with open('random_prompt.txt', 'w') as f:
    f.write(random_prompt)

with open('class_sorted_prompt.txt', 'w') as f:
    f.write(class_sorted_prompt)

with open('baseline_prompt.txt', 'w') as f:
    f.write(baseline_prompt)

print("Prompts have been saved to random_prompt.txt, class_sorted_prompt.txt, and baseline_prompt.txt.")