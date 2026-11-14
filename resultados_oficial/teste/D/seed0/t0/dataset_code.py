import numpy as np
import pandas as pd

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