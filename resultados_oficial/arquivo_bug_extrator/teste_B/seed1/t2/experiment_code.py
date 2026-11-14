from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import statistics

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

# Define a simple rule-based baseline (random guessing)
def baseline_predict(sentence):
    return torch.randint(0, 2, (1,)).item()

# Evaluate the baseline model
def evaluate_model(dataset, baseline=False):
    predictions = []
    true_labels = []
    for example in dataset:
        if baseline:
            prediction = baseline_predict(example['sentence'])
        else:
            # Simulate model prediction (in real code, replace with model inference)
            prediction = torch.randint(0, 2, (1,)).item()
        predictions.append(prediction)
        true_labels.append(example['label'])
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, f1

# Evaluate the baseline model on both versions
baseline_acc_a, baseline_f1_a = evaluate_model(test_dataset, baseline=True)
baseline_acc_b, baseline_f1_b = evaluate_model(test_dataset_b, baseline=True)

# Print baseline results
print("Baseline Results:")
print("Version A (Original sentences): Accuracy = {:.2f}, F1 = {:.2f}".format(baseline_acc_a, baseline_f1_a))
print("Version B (Negation words removed): Accuracy = {:.2f}, F1 = {:.2f}".format(baseline_acc_b, baseline_f1_b))

# Evaluate the language model
def evaluate_language_model(dataset):
    predictions = []
    true_labels = []
    for example in dataset:
        # Simulate model prediction (in real code, replace with model inference)
        prediction = torch.randint(0, 2, (1,)).item()
        predictions.append(prediction)
        true_labels.append(example['label'])
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, f1

# Evaluate the language model on both versions
model_acc_a, model_f1_a = evaluate_language_model(test_dataset)
model_acc_b, model_f1_b = evaluate_language_model(test_dataset_b)

# Print model results
print("Language Model Results:")
print("Version A (Original sentences): Accuracy = {:.2f}, F1 = {:.2f}".format(model_acc_a, model_f1_a))
print("Version B (Negation words removed): Accuracy = {:.2f}, F1 = {:.2f}".format(model_acc_b, model_f1_b))

# Compare performance between versions
diff_acc = model_acc_a - model_acc_b
diff_f1 = model_f1_a - model_f1_b

# Calculate standard error for t-test
std_error_acc = statistics.sqrt((model_acc_a*(1 - model_acc_a)/len(test_dataset) + model_acc_b*(1 - model_acc_b)/len(test_dataset)))
std_error_f1 = statistics.sqrt((model_f1_a*(1 - model_f1_a)/len(test_dataset) + model_f1_b*(1 - model_f1_b)/len(test_dataset)))

# Perform t-test
if std_error_acc != 0:
    t_statistic_acc = diff_acc / std_error_acc
else:
    t_statistic_acc = 0
if std_error_f1 != 0:
    t_statistic_f1 = diff_f1 / std_error_f1
else:
    t_statistic_f1 = 0

# Print statistical analysis
print("Statistical Analysis:")
print("Accuracy: t-statistic = {:.2f}".format(t_statistic_acc))
print("F1-score: t-statistic = {:.2f}".format(t_statistic_f1))