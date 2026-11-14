import os
import pandas as pd
from datasets import load_dataset

# Set your HF_TOKEN here if needed
# os.environ["HF_TOKEN"] = "your_hf_token_here"

# Load the SST-2 dataset with a small subset
sst2 = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Define negation words
negation_words = {'not', 'never', 'no', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}

# Split the dataset into two subsets: with and without negation words
with_negation = []
without_negation = []

for item in sst2:
    sentence = item['sentence']
    label = item['label']
    has_negation = any(word in sentence.lower() for word in negation_words)
    
    if has_negation:
        with_negation.append({"sentence": sentence, "label": label})
    else:
        without_negation.append({"sentence": sentence, "label": label})

# Convert subsets to pandas DataFrames for easier handling
with_negation_df = pd.DataFrame(with_negation)
without_negation_df = pd.DataFrame(without_negation)

# Display the first few items from each subset
print("Examples with negation words:")
print(with_negation_df.head())
print("\nExamples without negation words:")
print(without_negation_df.head())