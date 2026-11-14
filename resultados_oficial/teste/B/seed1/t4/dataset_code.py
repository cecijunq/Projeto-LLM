from sklearn.datasets import load_iris
import random

# Load the Iris dataset (a built-in sklearn dataset)
data = load_iris()
# Select 20 examples (10 per class) for few-shot learning
few_shot_data = data.data[:20]
few_shot_labels = data.target[:20]

# Create multiple permutations of the few-shot examples (e.g., 5 permutations)
num_permutations = 5
permutations = []
for _ in range(num_permutations):
    # Shuffle the examples
    shuffled_data = list(zip(few_shot_data, few_shot_labels))
    random.shuffle(shuffled_data)
    # Split into data and labels
    permuted_data = [item[0] for item in shuffled_data]
    permuted_labels = [item[1] for item in shuffled_data]
    permutations.append((permuted_data, permuted_labels))