import pandas as pd
from datasets import load_dataset

# Load the GSM8K dataset from HuggingFace
dataset = load_dataset('openai/gsm8k', 'main', split='test[:200]')

# Convert to a pandas DataFrame for easier handling
df = pd.DataFrame(dataset)

# Display the first few rows to verify the data
print(df.head())

# Save the dataset to a CSV file for later use
df.to_csv('gsm8k_subset.csv', index=False)