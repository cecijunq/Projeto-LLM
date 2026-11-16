import os
import random
from datasets import load_dataset
import json

# Load the IMDb dataset
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Filter to get 50 positive and 50 negative reviews
positive_reviews = [example for example in dataset if example['label'] == 1]
negative_reviews = [example for example in dataset if example['label'] == 0]

# Take 50 positive and 50 negative reviews (total 100)
positive_subset = positive_reviews[:50]
negative_subset = negative_reviews[:50]

# Combine into a list of examples
examples = positive_subset + negative_subset

# Split into few-shot examples (5 examples per class)
few_shot_positive = positive_subset[:5]
few_shot_negative = negative_subset[:5]

# Create three versions of the few-shot prompt
version_a = few_shot_positive + few_shot_negative  # Positive followed by negative
version_b = few_shot_negative + few_shot_positive  # Negative followed by positive
version_c = random.sample(few_shot_positive + few_shot_negative, len(few_shot_positive + few_shot_negative))  # Randomized order

# Define the test set (10 examples: 5 positive, 5 negative)
test_set = positive_subset[5:] + negative_subset[5:]

# Save the prompt versions and test set to files
with open('version_a.json', 'w') as f:
    json.dump([example['text'] for example in version_a], f)

with open('version_b.json', 'w') as f:
    json.dump([example['text'] for example in version_b], f)

with open('version_c.json', 'w') as f:
    json.dump([example['text'] for example in version_c], f)

with open('test_set.json', 'w') as f:
    json.dump([example['text'] for example in test_set], f)

print("Data preparation completed. Files saved: version_a.json, version_b.json, version_c.json, test_set.json")