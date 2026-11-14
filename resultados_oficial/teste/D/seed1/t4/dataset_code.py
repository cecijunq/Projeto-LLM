import os
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset('argilla/end2end_textclassification', split='train[:200]')

# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Print columns and an example
print("Columns:", df.columns)
print("Example:", df.iloc[0].to_dict())

# Extract text and label columns
text_column = 'text'
label_column = 'label'

# Check if the required columns exist
if text_column not in df.columns or label_column not in df.columns:
    raise ValueError(f"Dataset missing required columns: {text_column} or {label_column}")

# Prepare data for the experiment
print("Data preparation complete. Ready to proceed with the experiment.")