import pandas as pd
import numpy as np
from datasets import load_dataset
import random
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# Load synthetic-domain-text-classification dataset
dataset = load_dataset('argilla/synthetic-domain-text-classification', split='train[:200]')

# Ensure the dataset has the required columns
if 'text' not in dataset.column_names or 'label' not in dataset.column_names:
    raise ValueError("The dataset does not contain the required columns 'text' and 'label'.")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

# Separate positive and negative examples
positive_examples = df[df['label'] == 1]
negative_examples = df[df['label'] == 0]

# Select 2 examples per class
selected_positive = positive_examples.sample(n=2)
selected_negative = negative_examples.sample(n=2)

# Combine selected examples
few_shot_examples = pd.concat([selected_positive, selected_negative]).sample(frac=1).reset_index(drop=True)

# Generate permutations of few-shot examples
num_permutations = 10
permutations = [few_shot_examples.sample(frac=1).reset_index(drop=True) for _ in range(num_permutations)]

# Save permutations to a list of DataFrames for later use in prompting
permutation_list = permutations

# Output the number of permutations and examples
print(f"Generated {num_permutations} permutations with 4 few-shot examples each.")