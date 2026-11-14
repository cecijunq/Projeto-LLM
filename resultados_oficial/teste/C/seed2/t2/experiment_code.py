import json
import os
from openai import OpenAI

# Load processed data
with open('processed_data.json', 'r') as f:
    processed_data = json.load(f)

# Initialize the LLM client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Function to evaluate model response
def evaluate_model(prompt):
    try:
        r = client.chat.completions.create(
            model=os.environ['AGENTLAB_LLM_MODEL'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=64,
            temperature=0
        )
        answer = r.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return str(e)

# Function to check if response is a valid number
def is_valid_number(response):
    try:
        float(response)
        return True
    except ValueError:
        return False

# Evaluate original and negated texts
results = []

for i, example in enumerate(processed_data):
    if i >= 50:  # Limit to 50 examples
        break
    
    original_text = example['original_text']
    negated_text = example['negated_text']
    label = example['label']
    
    # Prompt for original text
    prompt_original = f"Classify the sentiment of the following review: '{original_text}'. Answer: 0 for negative, 1 for positive."
    response_original = evaluate_model(prompt_original)
    
    # Prompt for negated text
    prompt_negated = f"Classify the sentiment of the following review: '{negated_text}'. Answer: 0 for negative, 1 for positive."
    response_negated = evaluate_model(prompt_negated)
    
    # Check if responses are valid numbers
    valid_original = is_valid_number(response_original)
    valid_negated = is_valid_number(response_negated)
    
    # Convert to integers if valid
    predicted_label_original = int(response_original) if valid_original else -1
    predicted_label_negated = int(response_negated) if valid_negated else -1
    
    # Store results
    results.append({
        'original_text': original_text,
        'negated_text': negated_text,
        'label': label,
        'predicted_original': predicted_label_original,
        'predicted_negated': predicted_label_negated
    })

# Print results
print("Model predictions for original and negated texts:")
for result in results:
    print(f"Original: '{result['original_text']}' | Label: {result['label']} | Predicted: {result['predicted_original']}")
    print(f"Negated: '{result['negated_text']}' | Label: {result['label']} | Predicted: {result['predicted_negated']}\n")

# Calculate accuracy
correct_original = sum(1 for result in results if result['predicted_original'] == result['label'])
correct_negated = sum(1 for result in results if result['predicted_negated'] == result['label'])
total = len(results)

print(f"Accuracy on original texts: {correct_original}/{total} ({correct_original/total:.2f})")
print(f"Accuracy on negated texts: {correct_negated}/{total} ({correct_negated/total:.2f})")