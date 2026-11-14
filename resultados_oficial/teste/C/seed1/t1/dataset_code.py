import json
import requests
import time
import random

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
            "Reykjavik",
            "Helsinki",
            "Copenhagen"
        ],
        "correct_answer": "Oslo"
    },
    {
        "statement": "What is the capital of Ukraine?",
        "alternatives": [
            "Kyiv",
            "Oslo",
            "Reykjavik",
            "Helsinki"
        ],
        "correct_answer": "Kyiv"
    },
    {
        "statement": "Which city is the capital of Poland?",
        "alternatives": [
            "Warsaw",
            "Kyiv",
            "Oslo",
            "Reykjavik"
        ],
        "correct_answer": "Warsaw"
    },
    {
        "statement": "Which is the correct capital of Romania?",
        "alternatives": [
            "Bucharest",
            "Warsaw",
            "Kyiv",
            "Oslo"
        ],
        "correct_answer": "Bucharest"
    },
    {
        "statement": "What is the capital of Serbia?",
        "alternatives": [
            "Belgrade",
            "Bucharest",
            "Warsaw",
            "Kyiv"
        ],
        "correct_answer": "Belgrade"
    },
    {
        "statement": "Which city is the capital of Croatia?",
        "alternatives": [
            "Zagreb",
            "Belgrade",
            "Bucharest",
            "Warsaw"
        ],
        "correct_answer": "Zagreb"
    },
    {
        "statement": "Which is the correct capital of Slovenia?",
        "alternatives": [
            "Ljubljana",
            "Zagreb",
            "Belgrade",
            "Bucharest"
        ],
        "correct_answer": "Ljubljana"
    },
    {
        "statement": "What is the capital of Hungary?",
        "alternatives": [
            "Budapest",
            "Ljubljana",
            "Zagreb",
            "Belgrade"
        ],
        "correct_answer": "Budapest"
    }
]

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to send a prompt to Ollama and get the response
def query_ollama(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: " + str(response.status_code)

# Function to evaluate the model's answer
def evaluate_answer(question, model_answer):
    # Extract the answer from the model response
    model_answer = model_answer.strip()
    if model_answer in question["alternatives"]:
        return question["alternatives"].index(model_answer)
    else:
        return -1

# Main function to run the experiment
def run_experiment():
    results = []
    for question in questions:
        prompt = f"Question: {question['statement']}\nOptions:\n"
        for i, alt in enumerate(question["alternatives"]):
            prompt += f"{i+1}. {alt}\n"
        prompt += "Please select the correct option (1-4) and provide your answer in the format: Answer: [option]."
        
        model_response = query_ollama(prompt)
        model_answer = model_response.split("Answer: ")[-1].strip()
        answer_index = evaluate_answer(question, model_answer)
        
        results.append({
            "question": question,
            "model_answer": model_answer,
            "answer_index": answer_index,
            "correct_answer": question["correct_answer"]
        })
    
    # Calculate accuracy
    correct_answers = 0
    for result in results:
        if result["answer_index"] != -1 and result["alternatives"][result["answer_index"]] == result["correct_answer"]:
            correct_answers += 1
    
    accuracy = correct_answers / len(results)
    print(f"Model Accuracy: {accuracy:.2f} ({correct_answers}/{len(results)})")
    return results

# Run the experiment
run_experiment()