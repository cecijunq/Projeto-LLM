import pandas as pd
import random
import os
from openai import OpenAI

# Custom dataset: 20 multiple-choice questions with varying word orders
questions = [
    {
        "original": "Which planet is closest to the sun?",
        "altered": "To which planet is the sun closest?",
        "answer": "Mercury"
    },
    {
        "original": "What is the capital of France?",
        "altered": "Which city is the capital of France?",
        "answer": "Paris"
    },
    {
        "original": "Who wrote 'Pride and Prejudice'?",
        "altered": "Which author wrote 'Pride and Prejudice'?",
        "answer": "Jane Austen"
    },
    {
        "original": "Which element has the chemical symbol 'O'?",
        "altered": "What element has the chemical symbol 'O'?",
        "answer": "Oxygen"
    },
    {
        "original": "Which gas is most abundant in Earth's atmosphere?",
        "altered": "What gas is most abundant in Earth's atmosphere?",
        "answer": "Nitrogen"
    },
    {
        "original": "What is the largest planet in our solar system?",
        "altered": "Which planet is the largest in our solar system?",
        "answer": "Jupiter"
    },
    {
        "original": "Which country has the largest population?",
        "altered": "What country has the largest population?",
        "answer": "China"
    },
    {
        "original": "What is the primary language spoken in Brazil?",
        "altered": "Which language is primarily spoken in Brazil?",
        "answer": "Portuguese"
    },
    {
        "original": "Which element has the atomic number 1?",
        "altered": "What element has the atomic number 1?",
        "answer": "Hydrogen"
    },
    {
        "original": "Who is the author of 'To Kill a Mockingbird'?",
        "altered": "Which author wrote 'To Kill a Mockingbird'?",
        "answer": "Harper Lee"
    },
    {
        "original": "What is the boiling point of water at standard pressure?",
        "altered": "At standard pressure, what is the boiling point of water?",
        "answer": "100°C"
    },
    {
        "original": "Which planet is known as the Red Planet?",
        "altered": "What planet is known as the Red Planet?",
        "answer": "Mars"
    },
    {
        "original": "What is the largest ocean on Earth?",
        "altered": "Which ocean is the largest on Earth?",
        "answer": "Pacific Ocean"
    },
    {
        "original": "Which gas is most abundant in the Earth's atmosphere?",
        "altered": "What gas is most abundant in the Earth's atmosphere?",
        "answer": "Nitrogen"
    },
    {
        "original": "What is the full form of DNA?",
        "altered": "What does DNA stand for?",
        "answer": "Deoxyribonucleic Acid"
    },
    {
        "original": "Who painted the Mona Lisa?",
        "altered": "Which artist painted the Mona Lisa?",
        "answer": "Leonardo da Vinci"
    },
    {
        "original": "What is the largest mammal on Earth?",
        "altered": "Which mammal is the largest on Earth?",
        "answer": "Blue whale"
    },
    {
        "original": "What is the capital of Japan?",
        "altered": "Which city is the capital of Japan?",
        "answer": "Tokyo"
    },
    {
        "original": "Who is the current President of the United States?",
        "altered": "Which person is the current President of the United States?",
        "answer": "Joe Biden"
    },
    {
        "original": "What is the chemical symbol for carbon?",
        "altered": "What is the symbol for the element carbon?",
        "answer": "C"
    }
]

client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))
model_name = os.environ.get('AGENTLAB_LLM_MODEL', 'llama2')

# Function to get model answer
def get_model_answer(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Evaluate original and altered versions
original_accuracy = 0
altered_accuracy = 0
total_questions = len(questions)

for q in questions:
    # Original question
    prompt = q["original"]
    answer = get_model_answer(prompt)
    if answer == q["answer"]:
        original_accuracy += 1

    # Altered question
    prompt = q["altered"]
    answer = get_model_answer(prompt)
    if answer == q["answer"]:
        altered_accuracy += 1

original_accuracy /= total_questions
altered_accuracy /= total_questions

print("Original Accuracy:", original_accuracy)
print("Altered Accuracy:", altered_accuracy)