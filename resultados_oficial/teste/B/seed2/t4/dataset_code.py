from datasets import load_dataset

# Load the 'imdb' dataset with the specified split and columns
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Display the first few examples to confirm the data structure
print(dataset[:5])