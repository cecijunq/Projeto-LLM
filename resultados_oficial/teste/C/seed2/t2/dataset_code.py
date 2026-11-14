import datasets
import re

# Load the Kinopoisk sentiment classification dataset
dataset = datasets.load_dataset('ai-forever/kinopoisk-sentiment-classification', split='test[:200]')

# Function to remove negation words
def remove_negation_words(text):
    negation_words = ['not', 'never', 'no', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor']
    # Remove negation words and adjust punctuation
    for word in negation_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    # Remove extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Process the dataset to create original and negation-removed versions
processed_data = []
for example in dataset:
    original_text = example['text']
    negated_text = remove_negation_words(original_text)
    processed_data.append({
        'original_text': original_text,
        'negated_text': negated_text,
        'label': example['label']
    })

# Save the processed data to a JSON file
import json
with open('processed_data.json', 'w') as f:
    json.dump(processed_data, f)