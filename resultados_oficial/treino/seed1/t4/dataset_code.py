from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the breast_cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to introduce label noise
def introduce_label_noise(y, noise_percentage):
    noise_mask = np.random.rand(len(y)) < noise_percentage
    y_noisy = y.copy()
    y_noisy[noise_mask] = 1 - y_noisy[noise_mask]  # Flip labels
    return y_noisy

# Introduce label noise for different percentages
noise_percentages = [0.1, 0.2, 0.3]

# Store results
results = {}

for noise_level in noise_percentages:
    # Introduce noise
    y_train_noisy = introduce_label_noise(y_train, noise_level)
    
    # Train models
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train_noisy)
    
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train_noisy)
    
    # Evaluate on test set
    y_pred_logistic = logistic_model.predict(X_test)
    y_pred_gb = gb_model.predict(X_test)
    
    # Compute metrics
    acc_logistic = accuracy_score(y_test, y_pred_logistic)
    f1_logistic = f1_score(y_test, y_pred_logistic)
    
    acc_gb = accuracy_score(y_test, y_pred_gb)
    f1_gb = f1_score(y_test, y_pred_gb)
    
    # Store results
    results[noise_level] = {
        "logistic": {"accuracy": acc_logistic, "f1": f1_logistic},
        "gb": {"accuracy": acc_gb, "f1": f1_gb}
    }

# Print results
for noise_level, metrics in results.items():
    print(f"Noise Level: {noise_level * 100}%")
    print(f"Logistic Regression - Accuracy: {metrics['logistic']['accuracy']:.4f}, F1: {metrics['logistic']['f1']:.4f}")
    print(f"Gradient Boosting - Accuracy: {metrics['gb']['accuracy']:.4f}, F1: {metrics['gb']['f1']:.4f}")
    print("-" * 60)