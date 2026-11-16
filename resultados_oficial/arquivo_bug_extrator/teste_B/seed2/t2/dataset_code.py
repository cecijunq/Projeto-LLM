from datasets import load_dataset

# Load a small subset of the SST-2 dataset
dataset = load_dataset('stanfordnlp/sst2', split='test[:500]')

# Convert labels from -1 and 1 to 0 and 1 for binary classification
dataset = dataset.map(lambda x: {'label': 0 if x['label'] == -1 else 1, 'sentence': x['sentence']})

# Add negation words to the beginning or end of each review
negation_words = ['not', 'never', 'no']
dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label'], 'negated_sentence': x['sentence']})

for example in dataset:
    for negation_word in negation_words:
        example['negated_sentence'] = negation_word + ' ' + example['sentence']
        example['negated_sentence'] = example['sentence'] + ' ' + negation_word

# Remove any negation words present in the review
dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label'], 'negation_removed_sentence': x['sentence']})

for example in dataset:
    for negation_word in negation_words:
        example['negation_removed_sentence'] = example['negation_removed_sentence'].replace(negation_word, '')

# Display the first few examples to confirm the structure
print("Original Dataset:", dataset[:5])