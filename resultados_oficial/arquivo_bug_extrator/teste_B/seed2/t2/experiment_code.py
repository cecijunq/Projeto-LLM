from datasets import load_dataset
import os
from openai import OpenAI

# Load a small subset of the SST-2 dataset
dataset = load_dataset('stanfordnlp/sst2', split='test[:500]')

# Convert labels from -1 and 1 to 0 and 1 for binary classification
dataset = dataset.map(lambda x: {'label': 0 if x['label'] == -1 else 1, 'sentence': x['sentence']})

# Add negation words to the beginning or end of each review
negation_words = ['not', 'never', 'no']
dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label'], 'negated_sentence': x['sentence']})

for example in dataset:
    for negation_word in negation_words:
        example['negated_sentence'] = negation_word + ' ' + example['sentence']
        example['negated_sentence'] = example['sentence'] + ' ' + negation_word

# Remove any negation words present in the review
dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label'], 'negation_removed_sentence': x['sentence']})

for example in dataset:
    for negation_word in negation_words:
        example['negation_removed_sentence'] = example['negation_removed_sentence'].replace(negation_word, '')

# Display the first few examples to confirm the structure
print("Original Dataset:", dataset[:5])

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Function to get sentiment from the language model
def get_sentiment(sentence):
    response = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[
            {'role': 'user', 'content': f"What is the sentiment of the review '{sentence}'? Please respond with 'negative' or 'positive'."}
        ],
        max_tokens=64,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    if answer == 'negative':
        return 0
    elif answer == 'positive':
        return 1
    else:
        raise ValueError(f"Invalid sentiment response: {answer}")

# Evaluate the model's performance on the original dataset
original_accuracy = 0
original_f1 = 0
original_correct = 0
original_total = 0
original_true_positive = 0
original_true_negative = 0
original_false_positive = 0
original_false_negative = 0

for example in dataset:
    true_label = example['label']
    predicted_label = get_sentiment(example['sentence'])
    original_total += 1
    if predicted_label == true_label:
        original_correct += 1
    if predicted_label == 1 and true_label == 1:
        original_true_positive += 1
    elif predicted_label == 0 and true_label == 0:
        original_true_negative += 1
    elif predicted_label == 1 and true_label == 0:
        original_false_positive += 1
    elif predicted_label == 0 and true_label == 1:
        original_false_negative += 1

original_accuracy = original_correct / original_total
original_precision = original_true_positive / (original_true_positive + original_false_positive) if (original_true_positive + original_false_positive) > 0 else 0
original_recall = original_true_positive / (original_true_positive + original_false_negative) if (original_true_positive + original_false_negative) > 0 else 0
original_f1 = 2 * (original_precision * original_recall) / (original_precision + original_recall) if (original_precision + original_recall) > 0 else 0

print("Original Dataset Performance:")
print(f"Accuracy: {original_accuracy:.4f}")
print(f"F1 Score: {original_f1:.4f}")

# Evaluate the model's performance on the negated dataset
negated_accuracy = 0
negated_f1 = 0
negated_correct = 0
negated_total = 0
negated_true_positive = 0
negated_true_negative = 0
negated_false_positive = 0
negated_false_negative = 0

for example in dataset:
    true_label = example['label']
    predicted_label = get_sentiment(example['negated_sentence'])
    negated_total += 1
    if predicted_label == true_label:
        negated_correct += 1
    if predicted_label == 1 and true_label == 1:
        negated_true_positive += 1
    elif predicted_label == 0 and true_label == 0:
        negated_true_negative += 1
    elif predicted_label == 1 and true_label == 0:
        negated_false_positive += 1
    elif predicted_label == 0 and true_label == 1:
        negated_false_negative += 1

negated_accuracy = negated_correct / negated_total
negated_precision = negated_true_positive / (negated_true_positive + negated_false_positive) if (negated_true_positive + negated_false_positive) > 0 else 0
negated_recall = negated_true_positive / (negated_true_positive + negated_false_negative) if (negated_true_positive + negated_false_negative) > 0 else 0
negated_f1 = 2 * (negated_precision * negated_recall) / (negated_precision + negated_recall) if (negated_precision + negated_recall) > 0 else 0

print("Negated Dataset Performance:")
print(f"Accuracy: {negated_accuracy:.4f}")
print(f"F1 Score: {negated_f1:.4f}")

# Evaluate the model's performance on the negation-removed dataset
negation_removed_accuracy = 0
negation_removed_f1 = 0
negation_removed_correct = 0
negation_removed_total = 0
negation_removed_true_positive = 0
negation_removed_true_negative = 0
negation_removed_false_positive = 0
negation_removed_false_negative = 0

for example in dataset:
    true_label = example['label']
    predicted_label = get_sentiment(example['negation_removed_sentence'])
    negation_removed_total += 1
    if predicted_label == true_label:
        negation_removed_correct += 1
    if predicted_label == 1 and true_label == 1:
        negation_removed_true_positive += 1
    elif predicted_label == 0 and true_label == 0:
        negation_removed_true_negative += 1
    elif predicted_label == 1 and true_label == 0:
        negation_removed_false_positive += 1
    elif predicted_label == 0 and true_label == 1:
        negation_removed_false_negative += 1

negation_removed_accuracy = negation_removed_correct / negation_removed_total
negation_removed_precision = negation_removed_true_positive / (negation_removed_true_positive + negation_removed_false_positive) if (negation_removed_true_positive + negation_removed_false_positive) > 0 else 0
negation_removed_recall = negation_removed_true_positive / (negation_removed_true_positive + negation_removed_false_negative) if (negation_removed_true_positive + negation_removed_false_negative) > 0 else 0
negation_removed_f1 = 2 * (negation_removed_precision * negation_removed_recall) / (negation_removed_precision + negation_removed_recall) if (negation_removed_precision + negation_removed_recall) > 0 else 0

print("Negation-Removed Dataset Performance:")
print(f"Accuracy: {negation_removed_accuracy:.4f}")
print(f"F1 Score: {negation_removed_f1:.4f}")