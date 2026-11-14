import pandas as pd
from datasets import load_dataset

# Load dataset
dataset = load_dataset('mateus-hamade/multiple-choice-questions', split='train[:200]')

# Display column names and a sample entry
print("Columns:", dataset.column_names)
print("Sample entry:")
print(dataset[0])

# Convert to pandas DataFrame for easier handling
df = pd.DataFrame(dataset)

# Display first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())