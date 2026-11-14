import pandas as pd
from datasets import load_dataset

# Load the SST-2 dataset from HuggingFace
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Display the first few rows to verify the data
print(df.head())

# Save the dataset to a CSV file for later use
df.to_csv('sst2_subset.csv', index=False)