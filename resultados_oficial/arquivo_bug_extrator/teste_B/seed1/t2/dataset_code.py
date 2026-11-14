from datasets import load_dataset

# Load the SST-2 dataset (using a small subset for efficiency)
train_dataset = load_dataset('stanfordnlp/sst2', split='train[:200]')
test_dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Display the first few examples to confirm the structure
print("Original Dataset Examples:")
print("Train Dataset:")
print(train_dataset[:5])
print("Test Dataset:")
print(test_dataset[:5])

# Convert labels from -1 and 1 to 0 and 1
train_dataset = train_dataset.map(lambda x: {'label': 0 if x['label'] == -1 else 1})
test_dataset = test_dataset.map(lambda x: {'label': 0 if x['label'] == -1 else 1})

# Define negation words to remove
negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}

# Create Version B by removing negation words
def remove_negation_words(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in negation_words]
    return ' '.join(filtered_words)

test_dataset_b = test_dataset.map(lambda x: {'sentence': remove_negation_words(x['sentence'])})

# Display the first few examples of Version B
print("Test Dataset Version B Examples:")
print(test_dataset_b[:5])