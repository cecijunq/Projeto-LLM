import pandas as pd
from datasets import load_dataset
import random

# Load the IMDb dataset
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Convert the 'label' column to numerical values (0 for 'neg', 1 for 'pos')
dataset = dataset.map(lambda x: {'text': x['text'], 'label': 0 if x['label'] == 'neg' else 1})

# Separate the text and labels
texts = [example['text'] for example in dataset]
labels = [example['label'] for example in dataset]

# Select a fixed number of few-shot examples (e.g., 10)
num_few_shot = 10
few_shot_examples = random.sample(list(zip(texts, labels)), num_few_shot)
few_shot_texts, few_shot_labels = zip(*few_shot_examples)

# Function to shuffle the few-shot examples
def shuffle_few_shot(few_shot_texts, few_shot_labels):
    shuffled_examples = list(zip(few_shot_texts, few_shot_labels))
    random.shuffle(shuffled_examples)
    shuffled_texts, shuffled_labels = zip(*shuffled_examples)
    return shuffled_texts, shuffled_labels

# Generate multiple orderings of few-shot examples
num_orderings = 5
orderings = []
for _ in range(num_orderings):
    shuffled_texts, shuffled_labels = shuffle_few_shot(few_shot_texts, few_shot_labels)
    orderings.append((shuffled_texts, shuffled_labels))

# Control condition: no few-shot examples
control_condition = ((), ())  # Empty tuples to represent no examples

# Prepare data for model inference
inference_data = {
    'orderings': orderings,
    'control': control_condition
}

# Output the prepared data
inference_data