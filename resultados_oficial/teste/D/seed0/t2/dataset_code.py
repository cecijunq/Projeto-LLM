import pandas as pd
from datasets import load_dataset

# Load the SST-2 dataset with the test split and take the first 200 examples
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Check the columns and print one example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Format the data into a pandas DataFrame
df = pd.DataFrame(dataset)

# Ensure the columns are named 'question' and 'gold_answer' as required
# For SST-2, the 'sentence' column contains the text, and the 'label' column contains the sentiment (0 or 1)
# We will use 'sentence' as the question and 'label' as the gold answer
df.rename(columns={'sentence': 'question', 'label': 'gold_answer'}, inplace=True)

# Display the first few rows of the DataFrame to confirm the formatting
print(df.head())