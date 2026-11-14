import pandas as pd
from datasets import Dataset

# Define the list of 20 examples with original and manipulated versions
examples = [
    {
        "original_sentence": "The trophy doesn't fit in the box because it's too big.",
        "manipulated_sentence": "The trophy doesn't fit in the box because it's too small.",
        "options": ["The trophy", "The box"],
        "correct_answer": "The box"
    },
    {
        "original_sentence": "The flower is in the vase because it's beautiful.",
        "manipulated_sentence": "The flower is in the vase because it's ugly.",
        "options": ["The flower", "The vase"],
        "correct_answer": "The vase"
    },
    {
        "original_sentence": "The man saw the woman with the binoculars.",
        "manipulated_sentence": "The man saw the woman with the binoculars.",
        "options": ["The man", "The woman"],
        "correct_answer": "The woman"
    },
    {
        "original_sentence": "The woman saw the man with the binoculars.",
        "manipulated_sentence": "The woman saw the man with the binoculars.",
        "options": ["The woman", "The man"],
        "correct_answer": "The man"
    },
    {
        "original_sentence": "The cat is on the mat because it's comfortable.",
        "manipulated_sentence": "The cat is on the mat because it's uncomfortable.",
        "options": ["The cat", "The mat"],
        "correct_answer": "The mat"
    },
    {
        "original_sentence": "The dog is in the house because it's warm.",
        "manipulated_sentence": "The dog is in the house because it's cold.",
        "options": ["The dog", "The house"],
        "correct_answer": "The house"
    },
    {
        "original_sentence": "The child is playing with the toy because it's fun.",
        "manipulated_sentence": "The child is playing with the toy because it's boring.",
        "options": ["The child", "The toy"],
        "correct_answer": "The toy"
    },
    {
        "original_sentence": "The teacher is in the classroom because it's quiet.",
        "manipulated_sentence": "The teacher is in the classroom because it's noisy.",
        "options": ["The teacher", "The classroom"],
        "correct_answer": "The classroom"
    },
    {
        "original_sentence": "The chef is in the kitchen because it's clean.",
        "manipulated_sentence": "The chef is in the kitchen because it's messy.",
        "options": ["The chef", "The kitchen"],
        "correct_answer": "The kitchen"
    },
    {
        "original_sentence": "The student is in the library because it's quiet.",
        "manipulated_sentence": "The student is in the library because it's noisy.",
        "options": ["The student", "The library"],
        "correct_answer": "The library"
    },
    {
        "original_sentence": "The painter is in the studio because it's bright.",
        "manipulated_sentence": "The painter is in the studio because it's dark.",
        "options": ["The painter", "The studio"],
        "correct_answer": "The studio"
    },
    {
        "original_sentence": "The singer is in the concert because it's loud.",
        "manipulated_sentence": "The singer is in the concert because it's quiet.",
        "options": ["The singer", "The concert"],
        "correct_answer": "The concert"
    },
    {
        "original_sentence": "The baker is in the bakery because it's busy.",
        "manipulated_sentence": "The baker is in the bakery because it's quiet.",
        "options": ["The baker", "The bakery"],
        "correct_answer": "The bakery"
    },
    {
        "original_sentence": "The librarian is in the library because it's quiet.",
        "manipulated_sentence": "The librarian is in the library because it's noisy.",
        "options": ["The librarian", "The library"],
        "correct_answer": "The library"
    },
    {
        "original_sentence": "The driver is in the car because it's fast.",
        "manipulated_sentence": "The driver is in the car because it's slow.",
        "options": ["The driver", "The car"],
        "correct_answer": "The car"
    },
    {
        "original_sentence": "The painter is in the gallery because it's modern.",
        "manipulated_sentence": "The painter is in the gallery because it's traditional.",
        "options": ["The painter", "The gallery"],
        "correct_answer": "The gallery"
    },
    {
        "original_sentence": "The athlete is in the gym because it's clean.",
        "manipulated_sentence": "The athlete is in the gym because it's dirty.",
        "options": ["The athlete", "The gym"],
        "correct_answer": "The gym"
    },
    {
        "original_sentence": "The traveler is in the hotel because it's comfortable.",
        "manipulated_sentence": "The traveler is in the hotel because it's uncomfortable.",
        "options": ["The traveler", "The hotel"],
        "correct_answer": "The hotel"
    },
    {
        "original_sentence": "The chef is in the restaurant because it's busy.",
        "manipulated_sentence": "The chef is in the restaurant because it's quiet.",
        "options": ["The chef", "The restaurant"],
        "correct_answer": "The restaurant"
    },
    {
        "original_sentence": "The artist is in the studio because it's creative.",
        "manipulated_sentence": "The artist is in the studio because it's dull.",
        "options": ["The artist", "The studio"],
        "correct_answer": "The studio"
    }
]

# Convert to a Dataset
dataset = Dataset.from_dict({
    "original_sentence": [example["original_sentence"] for example in examples],
    "manipulated_sentence": [example["manipulated_sentence"] for example in examples],
    "options": [example["options"] for example in examples],
    "correct_answer": [example["correct_answer"] for example in examples]
})

import os
from openai import OpenAI

client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Function to evaluate the model on a single example
def evaluate_example(original_sentence, manipulated_sentence, options, correct_answer):
    # Query the model on the original sentence
    prompt_original = f"""
    You are a helpful assistant. You will be given a question with options. Choose the best answer based on the sentence.

    Sentence: {original_sentence}
    Options:
    1. {options[0]}
    2. {options[1]}

    Answer: 
    """
    r_original = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt_original}], max_tokens=64, temperature=0)
    answer_original = r_original.choices[0].message.content.strip()
    predicted_answer_original = options[0] if answer_original == "1." else options[1]
    correct = predicted_answer_original == correct_answer

    # Query the model on the manipulated sentence
    prompt_manipulated = f"""
    You are a helpful assistant. You will be given a question with options. Choose the best answer based on the sentence.

    Sentence: {manipulated_sentence}
    Options:
    1. {options[0]}
    2. {options[1]}

    Answer: 
    """
    r_manipulated = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt_manipulated}], max_tokens=64, temperature=0)
    answer_manipulated = r_manipulated.choices[0].message.content.strip()
    predicted_answer_manipulated = options[0] if answer_manipulated == "1." else options[1]
    correct_manipulated = predicted_answer_manipulated == correct_answer

    return correct, correct_manipulated

# Run the experiment
original_accuracy = 0
manipulated_accuracy = 0

for example in examples:
    correct, correct_manipulated = evaluate_example(
        example["original_sentence"],
        example["manipulated_sentence"],
        example["options"],
        example["correct_answer"]
    )
    original_accuracy += correct
    manipulated_accuracy += correct_manipulated

original_accuracy /= len(examples)
manipulated_accuracy /= len(examples)

print("Original Accuracy:", original_accuracy)
print("Manipulated Accuracy:", manipulated_accuracy)