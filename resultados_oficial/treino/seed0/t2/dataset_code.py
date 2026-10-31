import os
import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('azminetoushikwasi/math-story-problems', split='test[:200]')

# Extract questions and answers
questions = []
answers = []

for item in dataset:
    # Check if the item has the required columns
    if 'story_1_qs' in item and 'answer' in item:
        questions.append(item['story_1_qs'])
        answers.append(item['answer'])
    elif 'story_2_qs' in item and 'answer' in item:
        questions.append(item['story_2_qs'])
        answers.append(item['answer'])
    elif 'story_3_qs' in item and 'answer' in item:
        questions.append(item['story_3_qs'])
        answers.append(item['answer'])
    else:
        # Skip items that don't have the required columns
        continue

# Ensure we have exactly 100 questions and answers
questions = questions[:100]
answers = answers[:100]

# Save the questions and answers to a JSON file for later use
output_file = 'math_questions_answers.json'
with open(output_file, 'w') as f:
    json.dump({'questions': questions, 'answers': answers}, f)

print(f"Saved {len(questions)} questions and answers to {output_file}")