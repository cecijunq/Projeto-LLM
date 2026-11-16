import pandas as pd
import os
from openai import OpenAI

# Define a small hand-written list of 20 math questions with known answers
math_questions = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "Solve for x: 2x = 8", "answer": "4"},
    {"question": "What is the area of a square with side length 5?", "answer": "25"},
    {"question": "What is 10 divided by 2?", "answer": "5"},
    {"question": "Solve for x: 3x + 2 = 11", "answer": "3"},
    {"question": "What is the perimeter of a rectangle with length 6 and width 4?", "answer": "20"},
    {"question": "What is 3 squared?", "answer": "9"},
    {"question": "Solve for x: 5x - 3 = 12", "answer": "3"},
    {"question": "What is the volume of a cube with side length 3?", "answer": "27"},
    {"question": "What is 15 divided by 5?", "answer": "3"},
    {"question": "Solve for x: 4x + 1 = 13", "answer": "3"},
    {"question": "What is the area of a triangle with base 6 and height 4?", "answer": "12"},
    {"question": "What is 12 divided by 3?", "answer": "4"},
    {"question": "Solve for x: 2x + 5 = 15", "answer": "5"},
    {"question": "What is the perimeter of a square with side length 7?", "answer": "28"},
    {"question": "What is 8 squared?", "answer": "64"},
    {"question": "Solve for x: 6x - 2 = 16", "answer": "3"},
    {"question": "What is the volume of a rectangular prism with length 4, width 3, and height 2?", "answer": "24"},
    {"question": "What is 20 divided by 4?", "answer": "5"},
    {"question": "Solve for x: 7x + 1 = 22", "answer": "3"}
]

# Convert to DataFrame
df = pd.DataFrame(math_questions)

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Prompts for male and female role play
male_prompt = "You are a male student. Please solve this math problem."
female_prompt = "You are a female student. Please solve this math problem."

# Evaluate the model under both prompts
male_answers = []
female_answers = []

for index, row in df.iterrows():
    # Male prompt
    response = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{"role": "user", "content": male_prompt + " " + row['question']}],
        max_tokens=256,
        temperature=0
    )
    male_answer = response.choices[0].message.content.strip()
    male_answers.append((row['question'], row['answer'], male_answer))

    # Female prompt
    response = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{"role": "user", "content": female_prompt + " " + row['question']}],
        max_tokens=256,
        temperature=0
    )
    female_answer = response.choices[0].message.content.strip()
    female_answers.append((row['question'], row['answer'], female_answer))

# Calculate accuracy for male and female conditions
def calculate_accuracy(answers):
    correct = 0
    total = len(answers)
    for question, correct_answer, answer in answers:
        if answer.strip() == correct_answer.strip():
            correct += 1
    return correct / total if total > 0 else 0

male_accuracy = calculate_accuracy(male_answers)
female_accuracy = calculate_accuracy(female_answers)

# Print results
print("Male Accuracy:", male_accuracy)
print("Female Accuracy:", female_accuracy)