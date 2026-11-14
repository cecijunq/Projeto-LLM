from datasets import load_dataset
import random

# Load dataset
dataset = load_dataset("argilla/synthetic-domain-text-classification", split="train[:200]")

# Ensure the dataset has 'text' and 'label' columns
assert "text" in dataset.column_names, "Dataset must contain 'text' column"
assert "label" in dataset.column_names, "Dataset must contain 'label' column"

# Group examples by label
label_to_examples = {}
for example in dataset:
    text = example["text"]
    label = example["label"]
    if label not in label_to_examples:
        label_to_examples[label] = []
    label_to_examples[label].append(text)

# Shuffle examples within each label group
for label in label_to_examples:
    random.shuffle(label_to_examples[label])

# Flatten the examples into a single list
all_examples = []
for label in label_to_examples:
    all_examples.extend(label_to_examples[label])