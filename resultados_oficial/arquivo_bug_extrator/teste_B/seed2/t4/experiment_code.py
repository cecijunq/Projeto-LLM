from datasets import load_dataset
import random
import os
from openai import OpenAI

# Load the IMDB dataset with a small subset of 500 samples
dataset = load_dataset('stanfordnlp/imdb', 'plain_text', split='train[:500]')

# Split the dataset into training and test sets (80-20 split)
train_dataset = dataset.select(range(int(0.8 * len(dataset))))
test_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

# Select 5 examples from the training set for few-shot classification (2 positive, 3 negative)
positive_examples = [example for example in train_dataset if example['label'] == 1][:2]
negative_examples = [example for example in train_dataset if example['label'] == 0][:3]
few_shot_examples = positive_examples + negative_examples

# Shuffle the few-shot examples to create a random order
random_few_shot_examples = random.sample(few_shot_examples, len(few_shot_examples))

# Create the prompts
baseline_prompt = "Classify the following text as positive or negative:"

random_order_prompt = "Classify the following text as positive or negative:\n\n"
for example in random_few_shot_examples:
    random_order_prompt += f"Text: {example['text']}\nLabel: {example['label']}\n\n"

increasing_sentiment_prompt = "Classify the following text as positive or negative:\n\n"
for example in few_shot_examples:
    increasing_sentiment_prompt += f"Text: {example['text']}\nLabel: {example['label']}\n\n"

# Define the models to test
models = ['llama3', 'mixtral', 'llama2']

# Function to evaluate a model
def evaluate_model(model_name, prompt, test_dataset):
    client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))
    correct = 0
    total = 0
    
    for example in test_dataset:
        text = example['text']
        true_label = example['label']
        
        # Create the prompt for the current example
        full_prompt = prompt + f"\n\nText: {text}\nLabel: "
        
        # Query the model
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=64,
                temperature=0
            )
            
            # Extract the model's answer
            answer = response.choices[0].message.content.strip()
            
            # Check if the answer is correct
            if answer == '1' and true_label == 1:
                correct += 1
            elif answer == '0' and true_label == 0:
                correct += 1
            
            total += 1
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Evaluate the models with each prompt
print("Baseline Prompt Accuracy:")
for model in models:
    accuracy = evaluate_model(model, baseline_prompt, test_dataset)
    print(f"Model: {model}, Accuracy: {accuracy:.2f}")

print("\nRandom Order Prompt Accuracy:")
for model in models:
    accuracy = evaluate_model(model, random_order_prompt, test_dataset)
    print(f"Model: {model}, Accuracy: {accuracy:.2f}")

print("\nIncreasing Sentiment Order Prompt Accuracy:")
for model in models:
    accuracy = evaluate_model(model, increasing_sentiment_prompt, test_dataset)
    print(f"Model: {model}, Accuracy: {accuracy:.2f}")