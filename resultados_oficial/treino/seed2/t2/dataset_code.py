import pandas as pd

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
print(df.head())