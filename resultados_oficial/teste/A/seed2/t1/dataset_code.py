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
        "original_sentence": "The athlete is in the gym because it's clean.",
        "manipulated_sentence": "The athlete is in the gym because it's dirty.",
        "options": ["The athlete", "The gym"],
        "correct_answer": "The gym"
    },
    {
        "original_sentence": "The scientist is in the lab because it's quiet.",
        "manipulated_sentence": "The scientist is in the lab because it's noisy.",
        "options": ["The scientist", "The lab"],
        "correct_answer": "The lab"
    },
    {
        "original_sentence": "The cook is in the kitchen because it's warm.",
        "manipulated_sentence": "The cook is in the kitchen because it's cold.",
        "options": ["The cook", "The kitchen"],
        "correct_answer": "The kitchen"
    },
    {
        "original_sentence": "The driver is in the car because it's clean.",
        "manipulated_sentence": "The driver is in the car because it's dirty.",
        "options": ["The driver", "The car"],
        "correct_answer": "The car"
    },
    {
        "original_sentence": "The teacher is in the classroom because it's organized.",
        "manipulated_sentence": "The teacher is in the classroom because it's chaotic.",
        "options": ["The teacher", "The classroom"],
        "correct_answer": "The classroom"
    },
    {
        "original_sentence": "The gardener is in the garden because it's sunny.",
        "manipulated_sentence": "The gardener is in the garden because it's rainy.",
        "options": ["The gardener", "The garden"],
        "correct_answer": "The garden"
    },
    {
        "original_sentence": "The tourist is in the museum because it's interesting.",
        "manipulated_sentence": "The tourist is in the museum because it's boring.",
        "options": ["The tourist", "The museum"],
        "correct_answer": "The museum"
    },
    {
        "original_sentence": "The child is in the park because it's fun.",
        "manipulated_sentence": "The child is in the park because it's boring.",
        "options": ["The child", "The park"],
        "correct_answer": "The park"
    },
    {
        "original_sentence": "The artist is in the studio because it's creative.",
        "manipulated_sentence": "The artist is in the studio because it's uncreative.",
        "options": ["The artist", "The studio"],
        "correct_answer": "The studio"
    }
]

# Convert the list to a Dataset
dataset = Dataset.from_list(examples)

# Display the first few examples
print(dataset[0])