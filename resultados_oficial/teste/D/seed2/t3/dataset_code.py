import pandas as pd
from datasets import load_dataset

# Load the CIFAR-10 dataset from the verified source
dataset = load_dataset('uoft-cs/cifar10', 'plain_text')

# Check the columns and print a sample
print("Columns:", dataset['train'].column_names)
print("Sample data:", dataset['train'][0])

# Split the dataset into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Extract the first 200 examples from the test set
test_subset = test_dataset.select(range(200))

# Convert the dataset to a pandas DataFrame
train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_subset)

# Display the first few rows of the training and test DataFrames
print("Training DataFrame columns:", train_df.columns)
print("Training DataFrame sample:", train_df.head())
print("Test DataFrame columns:", test_df.columns)
print("Test DataFrame sample:", test_df.head())