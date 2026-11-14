import os
from datasets import load_dataset

# Load the SST-2 dataset with the correct configuration
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Print columns and a sample example
print("Columns:", dataset.column_names)
print("Sample example:", dataset[0])