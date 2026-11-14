import os
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Print columns and an example
print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# Extract text and label columns
texts = [example['text'] for example in dataset]
labels = [example['label'] for example in dataset]

# Print the first few texts and labels
print("\nFirst few texts and labels:")
for i in range(5):
    print(f"Text {i+1}: {texts[i]}")
    print(f"Label {i+1}: {labels[i]}\n")