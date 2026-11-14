import pandas as pd
from datasets import load_dataset
import random
import os
from openai import OpenAI

# Load the IMDb dataset
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Convert the 'label' column to numerical values (0 for 'neg', 1 for 'pos')
dataset = dataset.map(lambda x: {'text': x['text'], 'label': 0 if x['label'] == 'neg' else 1})

# Separate the text and labels
texts = [example['text'] for example in dataset]
labels = [example['label'] for example in dataset]

# Select a fixed number of few-shot examples (e.g., 10)
num_few_shot = 10
few_shot_examples = random.sample(list(zip(texts, labels)), num_few_shot)
few_shot_texts, few_shot_labels = zip(*few_shot_examples)

# Function to shuffle the few-shot examples
def shuffle_few_shot(few_shot_texts, few_shot_labels):
    shuffled_examples = list(zip(few_shot_texts, few_shot_labels))
    random.shuffle(shuffled_examples)
    shuffled_texts, shuffled_labels = zip(*shuffled_examples)
    return shuffled_texts, shuffled_labels

# Generate multiple orderings of few-shot examples
num_orderings = 5
orderings = []
for _ in range(num_orderings):
    shuffled_texts, shuffled_labels = shuffle_few_shot(few_shot_texts, few_shot_labels)
    orderings.append((shuffled_texts, shuffled_labels))

# Control condition: no few-shot examples
control_condition = ((), ())  # Empty tuples to represent no examples

# Prepare data for model inference
inference_data = {
    'orderings': orderings,
    'control': control_condition
}

client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))
def evaluate_model(condition, texts, labels):
    correct = 0
    total = 0
    for text, label in zip(texts, labels):
        prompt = f"Classify the following text as positive (1) or negative (0):\n{text}\nAnswer: "
        try:
            r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
            answer = r.choices[0].message.content.strip()
            if answer == '1':
                predicted_label = 1
            elif answer == '0':
                predicted_label = 0
            else:
                predicted_label = -1  # Invalid answer
            if predicted_label == label:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error for text: {text}, label: {label}, error: {str(e)}")
    accuracy = correct / total if total > 0 else 0
    return accuracy
    total = 0
    for text, label in zip(texts, labels):
        prompt = f"Classify the following text as positive (1) or negative (0):\n{text}\nAnswer: "
        try:
            r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
            answer = r.choices[0].message.content.strip()
            if answer == '1':
                predicted_label = 1
            elif answer == '0':
                predicted_label = 0
            else:
                predicted_label = -1  # Invalid answer
            if predicted_label == label:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error for text: {text}, label: {label}, error: {str(e)}")
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Run experiments
print("=== Experiment: Effect of Few-shot Example Order on Model Accuracy ===")
print("Control Condition (No Few-shot Examples):")
control_accuracy = evaluate_model(control_condition, texts, labels)
print(f"Accuracy: {control_accuracy:.2f}")

print("\n=== Few-shot Example Orderings ===")
for i, (texts, labels) in enumerate(orderings):
    print(f"Ordering {i+1}:")
    order_accuracy = evaluate_model((texts, labels), texts, labels)
    print(f"Accuracy: {order_accuracy:.2f}")