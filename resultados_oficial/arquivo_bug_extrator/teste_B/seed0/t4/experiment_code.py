from datasets import load_dataset
import random
import os
from openai import OpenAI

# Load dataset
dataset = load_dataset("argilla/synthetic-domain-text-classification", split="train[:200]")

# Ensure the dataset has 'text' and 'label' columns
assert "text" in dataset.column_names, "Dataset must contain 'text' column"
assert "label" in dataset.column_names, "Dataset must contain 'label' column"

# Group examples by label
label_to_examples = {}
for example in dataset:
    text = example["text"]
    label = example["label"]
    if label not in label_to_examples:
        label_to_examples[label] = []
    label_to_examples[label].append(text)

# Shuffle examples within each label group
for label in label_to_examples:
    random.shuffle(label_to_examples[label])

# Flatten the examples into a single list
all_examples = []
for label in label_to_examples:
    all_examples.extend(label_to_examples[label])

# Create a list of (text, label) pairs
example_label_pairs = []
for example in all_examples:
    # Find the corresponding label
    for label, texts in label_to_examples.items():
        if example in texts:
            example_label_pairs.append((example, label))
            break

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Set up the prompt format
prompt_template = "Examples:\n{examples}\n\nQuery: What is the category of this text? '{text}'\nAnswer:"

# Prepare the examples for each category
category_examples = {}
for label, texts in label_to_examples.items():
    category_examples[label] = texts

# Evaluate the model's performance for each category
for label, texts in category_examples.items():
    # Create a few-shot prompt with 3 examples
    if len(texts) >= 3:
        few_shot_examples = random.sample(texts, 3)
        examples_str = "\n".join([f"{text} - {label}" for text in few_shot_examples])
        prompt = prompt_template.format(examples=examples_str, text=texts[0])
        
        # Send the request to the model
        r = client.chat.completions.create(
            model=os.environ['AGENTLAB_LLM_MODEL'],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0
        )
        
        # Get the model's answer
        answer = r.choices[0].message.content.strip()
        
        # Check if the answer matches the label
        if answer == label:
            accuracy = 1.0
        else:
            accuracy = 0.0
        
        # Print the result
        print(f"Category: {label}")
        print(f"Examples: {examples_str}")
        print(f"Model's answer: {answer}")
        print(f"Accuracy for this category: {accuracy}")
        print("-" * 50)
    else:
        print(f"Category: {label}")
        print(f"Examples: {texts}")
        print("Model's answer: Not enough examples to evaluate.")
        print("Accuracy for this category: 0.0")
        print("-" * 50)