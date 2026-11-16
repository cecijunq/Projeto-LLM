import pandas as pd
from datasets import load_dataset

# Load the GSM8K dataset with the 'main' configuration and split into test set
dataset = load_dataset('openai/gsm8k', 'main', split='test[:200]')

# Convert to a pandas DataFrame for easier handling
df = dataset.to_pandas()

# Display the first few rows to confirm the data structure
print(df.head())

# Save the dataset to a CSV file for further processing
df.to_csv('gsm8k_test_data.csv', index=False)