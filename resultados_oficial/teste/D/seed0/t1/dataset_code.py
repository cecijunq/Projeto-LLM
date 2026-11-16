import pandas as pd
from datasets import load_dataset

# Load the Diabetes multiple-choice questions dataset
dataset = load_dataset('WaltonFuture/Diabetes-multiple-choice-questions', split='train[:200]')

# Check the columns and print an example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Ensure the dataset has the required columns: question and answer
if 'problem' in dataset.column_names and 'answer' in dataset.column_names:
    # Prepare the data for the experiment
    df = pd.DataFrame(dataset)
    print("Data preparation completed.")
else:
    print("The dataset does not have the required columns 'question' and 'answer'.")