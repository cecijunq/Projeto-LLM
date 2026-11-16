import numpy as np
import pandas as pd
import random
import os
from openai import OpenAI

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Hand-written examples for the topic "Are image transformers more or less sensitive to pixel noise than convolutional networks?"
examples = [
    {"image": np.random.rand(32, 32, 3), "label": 0, "noisy_image": np.random.rand(32, 32, 3), "noise_level": 0},
    {"image": np.random.rand(32, 32, 3), "label": 1, "noisy_image": np.random.rand(32, 32, 3), "noise_level": 0.2},
    {"image": np.random.rand(32, 32, 3), "label": 2, "noisy_image": np.random.rand(32, 32, 3), "noise_level": 0.4},
    {"image": np.random.rand(32, 32, 3), "label": 3, "noisy_image": np.random.rand(32, 32, 3), "noise_level": 0.6},
    {"image": np.random.rand(32, 32, 3), "label": 4, "noisy_image": np.random.rand(32, 32, 3), "noise_level": 0.8},
    # Add more examples as needed
]

# Convert to DataFrame
df = pd.DataFrame(examples)
print(df.columns)
print(df.head(1))

# Assume accuracy_results is a dictionary with noise levels as keys and accuracy as values
accuracy_results = {
    "noise_level_0": 0.95,
    "noise_level_0.2": 0.88,
    "noise_level_0.4": 0.76,
    "noise_level_0.6": 0.62,
    "noise_level_0.8": 0.45
}

# Print the accuracy results
for condition, accuracy in accuracy_results.items():
    print(f"Accuracy ({condition}): {accuracy:.4f}")