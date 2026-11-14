import pandas as pd
import random

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
        "answer": "Blue Whale"
    },
    {
        "original": "Which planet is known for its rings?",
        "altered": "What planet is known for its rings?",
        "answer": "Saturn"
    },
    {
        "original": "What is the capital of Japan?",
        "altered": "Which city is the capital of Japan?",
        "answer": "Tokyo"
    },
    {
        "original": "Which element has the chemical symbol 'Fe'?",
        "altered": "What element has the chemical symbol 'Fe'?",
        "answer": "Iron"
    }
]

# Convert to DataFrame for easier handling
df = pd.DataFrame(questions)

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())