from datasets import load_dataset

# Load the IMDB dataset with a small subset of 500 samples
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='train[:500]')

# Split the dataset into training and test sets (80-20 split)
train_dataset = dataset.select(range(int(0.8 * len(dataset))))
test_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

# Select 5 examples from the training set for few-shot classification (2 positive, 3 negative)
positive_examples = [example for example in train_dataset if example['label'] == 1][:2]
negative_examples = [example for example in train_dataset if example['label'] == 0][:3]
few_shot_examples = positive_examples + negative_examples

# Display the selected few-shot examples
few_shot_examples