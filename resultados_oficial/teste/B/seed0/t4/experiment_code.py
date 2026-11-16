import os
import random
from openai import OpenAI
import json

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Load the IMDb dataset
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='test[:200]')

# Filter to get 50 positive and 50 negative reviews
positive_reviews = [example for example in dataset if example['label'] == 1]
negative_reviews = [example for example in dataset if example['label'] == 0]

# Take 50 positive and 50 negative reviews (total 100)
positive_subset = positive_reviews[:50]
negative_subset = negative_reviews[:50]

# Combine into a list of examples
examples = positive_subset + negative_subset

# Split into few-shot examples (5 examples per class)
few_shot_positive = positive_subset[:5]
few_shot_negative = negative_subset[:5]

# Create three versions of the few-shot prompt
version_a = few_shot_positive + few_shot_negative  # Positive followed by negative
version_b = few_shot_negative + few_shot_positive  # Negative followed by positive
version_c = random.sample(few_shot_positive + few_shot_negative, len(few_shot_positive + few_shot_negative))  # Randomized order

# Define the test set (10 examples: 5 positive, 5 negative)
test_set = positive_subset[5:] + negative_subset[5:]

# Save the prompt versions and test set to files
with open('version_a.json', 'w') as f:
    json.dump([example['text'] for example in version_a], f)

with open('version_b.json', 'w') as f:
    json.dump([example['text'] for example in version_b], f)

with open('version_c.json', 'w') as f:
    json.dump([example['text'] for example in version_c], f)

with open('test_set.json', 'w') as f:
    json.dump([example['text'] for example in test_set], f)

print("Data preparation completed. Files saved: version_a.json, version_b.json, version_c.json, test_set.json")

# Load the prompt versions and test set
with open('version_a.json', 'r') as f:
    version_a_texts = json.load(f)
with open('version_b.json', 'r') as f:
    version_b_texts = json.load(f)
with open('version_c.json', 'r') as f:
    version_c_texts = json.load(f)
with open('test_set.json', 'r') as f:
    test_set_texts = json.load(f)

# Function to evaluate a prompt version
def evaluate_prompt(prompt_texts, test_texts):
    correct = 0
    total = len(test_texts)
    for idx, text in enumerate(test_texts):
        prompt = "\n".join(prompt_texts) + "\n\n" + text
        response = client.chat.completions.create(
            model=os.environ['AGENTLAB_LLM_MODEL'],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        # Determine the label from the answer
        if "positive" in answer or "pos" in answer or "good" in answer:
            predicted_label = 1
        elif "negative" in answer or "neg" in answer or "bad" in answer:
            predicted_label = 0
        else:
            predicted_label = -1
        # Determine the true label
        true_label = 1 if "positive" in test_texts[idx] else 0
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / total
    return accuracy

# Evaluate each version
version_a_accuracy = evaluate_prompt(version_a_texts, test_set_texts)
version_b_accuracy = evaluate_prompt(version_b_texts, test_set_texts)
version_c_accuracy = evaluate_prompt(version_c_texts, test_set_texts)

# Print results
print(f"Accuracy for Version A (Positive first): {version_a_accuracy:.2f}")
print(f"Accuracy for Version B (Negative first): {version_b_accuracy:.2f}")
print(f"Accuracy for Version C (Randomized order): {version_c_accuracy:.2f}")