import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Subsample 1000 samples (the dataset is small, so no subsampling needed)
# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# Function to introduce label noise
def introduce_label_noise(y, noise_rate):
    noisy_y = y.copy()
    noise_mask = np.random.rand(len(noisy_y)) < noise_rate
    noisy_y[noise_mask] = np.random.permutation(noisy_y)[noise_mask]
    return noisy_y

# Experimental parameters
noise_rates = [0.1, 0.2]
num_runs = 5

# Results storage
results = {'noise_rate': [], 'model': [], 'accuracy': []}

for run in range(num_runs):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + run)
    
    for noise_rate in noise_rates:
        # Introduce label noise
        noisy_y_train = introduce_label_noise(y_train, noise_rate)
        
        # Train logistic regression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, noisy_y_train)
        lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
        
        # Train gradient boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42 + run)
        gb_model.fit(X_train, noisy_y_train)
        gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))
        
        # Store results
        results['noise_rate'].extend([noise_rate, noise_rate])
        results['model'].extend(['Logistic Regression', 'Gradient Boosting'])
        results['accuracy'].extend([lr_accuracy, gb_accuracy])

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Print average accuracy for each noise rate and model
print("Average Accuracy Results:")
print(results_df.groupby(['noise_rate', 'model'])['accuracy'].mean().unstack())