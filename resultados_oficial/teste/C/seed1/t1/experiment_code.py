import json
import requests
import time
import random
import os
from openai import OpenAI

# Define the questions with variations in word order
questions = [
    {
        "statement": "Which is the correct capital of France?",
        "alternatives": [
            "Paris",
            "London",
            "Berlin",
            "Madrid"
        ],
        "correct_answer": "Paris"
    },
    {
        "statement": "What is the capital of Germany?",
        "alternatives": [
            "Berlin",
            "Paris",
            "Madrid",
            "London"
        ],
        "correct_answer": "Berlin"
    },
    {
        "statement": "Which city is the capital of Spain?",
        "alternatives": [
            "Madrid",
            "Berlin",
            "Paris",
            "London"
        ],
        "correct_answer": "Madrid"
    },
    {
        "statement": "Which is the correct capital of Italy?",
        "alternatives": [
            "Rome",
            "Madrid",
            "Berlin",
            "Paris"
        ],
        "correct_answer": "Rome"
    },
    {
        "statement": "What is the capital of Portugal?",
        "alternatives": [
            "Lisbon",
            "Rome",
            "Madrid",
            "Berlin"
        ],
        "correct_answer": "Lisbon"
    },
    {
        "statement": "Which city is the capital of Belgium?",
        "alternatives": [
            "Brussels",
            "Lisbon",
            "Rome",
            "Madrid"
        ],
        "correct_answer": "Brussels"
    },
    {
        "statement": "Which is the correct capital of Switzerland?",
        "alternatives": [
            "Bern",
            "Brussels",
            "Lisbon",
            "Rome"
        ],
        "correct_answer": "Bern"
    },
    {
        "statement": "What is the capital of Norway?",
        "alternatives": [
            "Oslo",
            "Bern",
            "Brussels",
            "Lisbon"
        ],
        "correct_answer": "Oslo"
    },
    {
        "statement": "Which city is the capital of Sweden?",
        "alternatives": [
            "Stockholm",
            "Oslo",
            "Bern",
            "Brussels"
        ],
        "correct_answer": "Stockholm"
    },
    {
        "statement": "Which is the correct capital of Denmark?",
        "alternatives": [
            "Copenhagen",
            "Stockholm",
            "Oslo",
            "Bern"
        ],
        "correct_answer": "Copenhagen"
    },
    {
        "statement": "What is the capital of Finland?",
        "alternatives": [
            "Helsinki",
            "Copenhagen",
            "Stockholm",
            "Oslo"
        ],
        "correct_answer": "Helsinki"
    },
    {
        "statement": "Which city is the capital of Iceland?",
        "alternatives": [
            "Reykjavik",
            "Helsinki",
            "Copenhagen",
            "Stockholm"
        ],
        "correct_answer": "Reykjavik"
    },
    {
        "statement": "Which is the correct capital of Norway?",
        "alternatives": [
            "Oslo",
            "Bern",
            "Brussels",
            "Lisbon"
        ],
        "correct_answer": "Oslo"
    }
]

# Set up API parameters
base_url = os.environ.get('AGENTLAB_LLM_BASE_URL')
api_key = os.environ.get('OPENAI_API_KEY', 'x')
model = os.environ.get('AGENTLAB_LLM_MODEL')

# Function to call the model
def call_model(prompt):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0
    )
    return response.choices[0].message.content

# Run the experiment
results = []
for question in questions:
    statement = question["statement"]
    alternatives = question["alternatives"]
    correct_answer = question["correct_answer"]
    
    # Create a prompt with the question and alternatives
    prompt = f"{statement}\nOptions:\n- {alternatives[0]}\n- {alternatives[1]}\n- {alternatives[2]}\n- {alternatives[3]}\nAnswer:"
    
    # Call the model
    answer = call_model(prompt)
    
    # Determine if the model's answer is correct
    is_correct = answer.strip() == correct_answer
    
    # Store the result
    results.append({
        "question": statement,
        "alternatives": alternatives,
        "correct_answer": correct_answer,
        "model_answer": answer.strip(),
        "is_correct": is_correct
    })

# Calculate accuracy
correct_answers = sum(result["is_correct"] for result in results)
total_questions = len(results)
accuracy = correct_answers / total_questions

# Print results
print(f"Model Accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
print("\n=== experiment output ===")
for result in results:
    print(f"Question: {result['question']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Is Correct: {result['is_correct']}")
    print("-" * 50)