from datasets import load_dataset
import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Use a small hand-written list of examples for the experiment
examples = [
    {
        "question": "The apple is red. The banana is yellow. Which fruit is red?",
        "options": ["apple", "banana", "orange", "grape"],
        "correct_answer": "apple"
    },
    {
        "question": "The cat is on the mat. The dog is under the mat. Which animal is on the mat?",
        "options": ["cat", "dog", "bird", "fish"],
        "correct_answer": "cat"
    },
    {
        "question": "John gave Mary a gift. Mary gave John a gift. Who gave a gift to whom?",
        "options": ["John gave a gift to Mary", "Mary gave a gift to John", "John gave a gift to John", "Mary gave a gift to Mary"],
        "correct_answer": "Mary gave a gift to John"
    },
    {
        "question": "The boy is playing with a ball. The ball is red. What is the color of the ball?",
        "options": ["red", "blue", "green", "yellow"],
        "correct_answer": "red"
    },
    {
        "question": "The girl is wearing a blue dress. The dress is long. What is the color of the dress?",
        "options": ["blue", "red", "green", "yellow"],
        "correct_answer": "blue"
    }
]

# Generate prompts for the language model based on the structured examples
prompts = []
for example in examples:
    question = example['question']
    options = example['options']
    correct_answer = example['correct_answer']
    prompt = f"Question: {question}\nOptions: {options}\nAnswer:"
    prompts.append(prompt)

# Evaluate the language model's performance
correct_answers = 0
total_questions = len(examples)

for i, example in enumerate(examples):
    question = example['question']
    options = example['options']
    correct_answer = example['correct_answer']
    
    # Call the language model to get the answer
    r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompts[i]}], max_tokens=64, temperature=0)
    model_answer = r.choices[0].message.content.strip()
    
    # Extract the model's answer from the response
    if model_answer.startswith("Answer: "):
        model_answer = model_answer[len("Answer: "):].strip()
    elif model_answer.startswith("Answer:"):
        model_answer = model_answer[len("Answer:"):].strip()
    
    # Check if the model's answer matches the correct answer
    is_correct = model_answer == correct_answer
    
    # Print the results
    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Model Answer: {model_answer}")
    print(f"Is Correct: {is_correct}")
    print("-" * 40)
    
    # Update the correct answers count
    if is_correct:
        correct_answers += 1

# Print the final accuracy
print(f"Accuracy: {correct_answers}/{total_questions} ({correct_answers / total_questions * 100:.2f}%)")