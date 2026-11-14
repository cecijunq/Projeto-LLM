from datasets import load_dataset
import random

# Load the MMLU dataset with a subset of 200 questions from the 'test' split
dataset = load_dataset('cais/mmlu', 'all', split='test[:200]')

# Check if the required columns are present
required_columns = ['question', 'subject', 'choices', 'answer']
missing_columns = [col for col in required_columns if col not in dataset.column_names]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

# Shuffle the order of answer choices for each question
shuffled_dataset = dataset.map(lambda example: {
    'question': example['question'],
    'subject': example['subject'],
    'choices': random.sample(example['choices'], len(example['choices'])),
    'answer': example['answer']
})

# Save the shuffled dataset to a temporary file
shuffled_dataset.save_to_disk("shuffled_dataset")