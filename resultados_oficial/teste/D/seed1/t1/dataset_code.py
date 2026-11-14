from datasets import load_dataset

# Load the dataset
dataset = load_dataset('WaltonFuture/Diabetes-multiple-choice-questions', split='train[:200]')

# Check column names and a sample
print("Column names:", dataset.column_names)
print("Sample:", dataset[0])

# Extract the question and answer
questions = [example['problem'] for example in dataset]
answers = [example['answer'] for example in dataset]

# Prepare the data for the plan
print("Questions:", questions)
print("Answers:", answers)