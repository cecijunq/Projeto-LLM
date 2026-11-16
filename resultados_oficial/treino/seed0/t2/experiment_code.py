import os
from openai import OpenAI
import json

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

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

# Ensure we have exactly 50 questions and answers
questions = questions[:50]
answers = answers[:50]

# Save the questions and answers to a JSON file for later use
output_file = 'math_questions_answers.json'
with open(output_file, 'w') as f:
    json.dump({'questions': questions, 'answers': answers}, f)

print(f"Saved {len(questions)} questions and answers to {output_file}")

# Evaluate model performance with neutral and gender-role prompts
neutral_prompts = [f"Q: {q}\nA:" for q in questions]
gender_prompts = [f"Q: A man is solving this math problem. {q}\nA:" for q in questions]

# Store results
neutral_results = []
gender_results = []

# Evaluate with neutral prompts
for prompt in neutral_prompts[:50]:
    r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
    answer = r.choices[0].message.content.strip()
    neutral_results.append(answer)

# Evaluate with gender-role prompts
for prompt in gender_prompts[:50]:
    r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)
    answer = r.choices[0].message.content.strip()
    gender_results.append(answer)

# Evaluate accuracy
neutral_answers = [a.split("A:")[1].strip() if "A:" in a else a for a in neutral_results]
gender_answers = [a.split("A:")[1].strip() if "A:" in a else a for a in gender_results]

correct_neutral = sum(1 for i in range(50) if neutral_answers[i] == answers[i])
correct_gender = sum(1 for i in range(50) if gender_answers[i] == answers[i])

print(f"Accuracy with neutral prompts: {correct_neutral}/{50}")
print(f"Accuracy with gender-role prompts: {correct_gender}/{50}")