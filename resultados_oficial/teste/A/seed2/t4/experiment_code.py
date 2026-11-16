import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Create a small hand-written dataset for text classification
# 5 classes with 5 examples each (total 25 examples)

# Define the dataset
dataset = {
    "Class 1": [
        "This movie was incredible! The acting was top-notch and the plot was engaging.",
        "The film was a masterpiece. It captured the essence of the story perfectly.",
        "A must-see for anyone who loves cinema. The direction was brilliant.",
        "I was completely captivated by the performances and the cinematography.",
        "The soundtrack added an emotional depth that elevated the overall experience."
    ],
    "Class 2": [
        "The food was terrible. It was overpriced and the taste was poor.",
        "I would not recommend this restaurant. The service was slow and the ambiance was unpleasant.",
        "The menu lacked variety, and the portions were way too small for the price.",
        "The staff was unhelpful and the overall experience was disappointing.",
        "The quality of the food did not match the high ratings on the website."
    ],
    "Class 3": [
        "This product exceeded my expectations. It works perfectly and is worth every penny.",
        "The packaging was well-protected, and the item arrived in perfect condition.",
        "I am extremely satisfied with the performance and durability of this product.",
        "The customer support was prompt and helpful in resolving any issues.",
        "This is one of the best purchases I've made in a long time."
    ],
    "Class 2": [
        "I am extremely disappointed with this product. It broke after just a few uses.",
        "The quality is subpar, and it doesn't deliver the promised results.",
        "The customer service was unresponsive and didn't provide a satisfactory solution.",
        "The delivery was delayed, and the item arrived damaged.",
        "I would not recommend this product to anyone."
    ],
    "Class 5": [
        "This is an amazing piece of art. The creativity and detail are beyond impressive.",
        "The artist's vision is evident in every brushstroke, and the composition is stunning.",
        "The use of color and texture adds a unique dimension to the work.",
        "This art piece is a must-have for any collector or art enthusiast.",
        "It has a timeless quality that continues to inspire and captivate viewers."
    ]
}

# Convert the dataset into a list of dictionaries
examples = []
for label, texts in dataset.items():
    for text in texts:
        examples.append({"text": text, "label": label})

# Split the dataset into training and test sets
train_examples, test_examples = train_test_split(examples, test_size=0.2, stratify=[e["label"] for e in examples], random_state=42)

# Convert the examples into pandas DataFrames for easier handling
train_df = pd.DataFrame(train_examples)
test_df = pd.DataFrame(test_examples)

import os
from openai import OpenAI

# Initialize the LLM client
client = OpenAI(
    base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'),
    api_key=os.environ.get('OPENAI_API_KEY', 'x')
)

# Function to generate predictions using the LLM
def predict_with_model(prompt):
    response = client.chat.completions.create(
        model=os.environ.get('AGENTLAB_LLM_MODEL'),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0
    )
    return response.choices[0].message.content

# Function to evaluate the accuracy of the model
def evaluate_model(prompt, test_examples):
    correct = 0
    total = len(test_examples)
    for example in test_examples:
        prompt_for_example = f"Classify the following text into one of the 5 classes: Class 1, Class 2, Class 3, Class 4, or Class 5.\n\n{example['text']}\nAnswer:"
        answer = predict_with_model(prompt_for_example)
        if answer.strip() == example['label']:
            correct += 1
    accuracy = correct / total * 100
    return accuracy

# Function to create a few-shot prompt with examples in a specific order
def create_few_shot_prompt(examples, num_examples=5):
    prompt = "Classify the following text into one of the 5 classes: Class 1, Class 2, Class 3, Class 4, or Class 5.\n\n"
    for i in range(num_examples):
        text = examples[i]["text"]
        label = examples[i]["label"]
        prompt += f"Text: {text}\nLabel: {label}\n\n"
    return prompt

# List of example orders to test
example_orders = [
    random.sample(train_examples, len(train_examples)),
    random.sample(train_examples, len(train_examples)),
    random.sample(train_examples, len(train_examples)),
    random.sample(train_examples, len(train_examples)),
    random.sample(train_examples, len(train_examples)),
]

# Run experiments
print("Testing the impact of few-shot example order on model accuracy...")
for order in example_orders:
    prompt = create_few_shot_prompt(order, num_examples=5)
    accuracy = evaluate_model(prompt, test_examples)
    print(f"Accuracy with this example order: {accuracy:.2f}%")