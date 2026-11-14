import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = load_digits()
train_data, test_data, train_labels, test_labels = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Generate synthetic noisy digits data
def add_gaussian_noise(images, noise_level):
    noise = np.random.normal(0, noise_level, images.shape)
    noisy_image = images + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

# Define noise levels
noise_levels = [0, 0.1, 0.2, 0.3, 0.4]

# Create a DataFrame to store the data
data = []

# Add clean and noisy data
for noise_level in noise_levels:
    noisy_data = add_gaussian_noise(train_data, noise_level)
    for i in range(len(train_data)):
        data.append({
            'image': noisy_data[i].flatten(),  # Flatten image for easier processing
            'label': train_labels[i]
        })

# Convert to DataFrame
df = pd.DataFrame(data)
print("Columns:", df.columns)
print("Example:", df.iloc[0])