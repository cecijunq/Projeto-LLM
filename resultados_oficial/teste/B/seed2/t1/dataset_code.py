from datasets import load_dataset

# Load the 'WaltonFuture/Diabetes-multiple-choice-questions' dataset with a small subset
diabetes_dataset = load_dataset('WaltonFuture/Diabetes-multiple-choice-questions', split='train[:200]')

# Structure the dataset to include problem, options, and correct answer
structured_diabetes_data = diabetes_dataset.map(lambda x: {
    'question': x['problem'],
    'options': [option.strip() for option in x['problem'].split(')') if option.strip() != ''],
    'correct_answer': x['answer']
})

# Display the first few entries of the structured data
structured_diabetes_data[:5]

# Verify the correctness of the structured data by checking if the correct answer exists in the options
verification_results = []
for i in range(len(structured_diabetes_data)):
    question = structured_diabetes_data['question'][i]
    options_list = structured_diabetes_data['options'][i]
    correct_answer = structured_diabetes_data['correct_answer'][i]
    is_correct = correct_answer in options_list
    verification_results.append({
        'question': question,
        'options': options_list,
        'correct_answer': correct_answer,
        'is_correct': is_correct
    })

# Display the first few verification results
verification_results[:5]

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

# Display the generated prompts
prompts