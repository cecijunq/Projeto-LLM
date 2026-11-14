import pandas as pd
from datasets import load_dataset

# Load the SST-2 dataset and select a test split of at least 200 examples
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Check the columns and print one example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Display the first few rows to confirm the structure
print("First few rows of the dataset:")
print(df.head())