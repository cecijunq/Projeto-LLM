from datasets import load_dataset

# Load the MedQA-USMLE-4-options dataset with a small subset
dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test[:200]')

# Display the first few examples to verify the structure
print("First few examples of the dataset:")
print(dataset[:5])

# Extract the relevant columns for the evaluation
questions = [example['question'] for example in dataset]
options = [example['options'] for example in dataset]
correct_answers = [example['answer'] for example in dataset]

# Display the first few questions, options, and correct answers
print("\nFirst few questions:")
print(questions[:5])

print("\nFirst few options:")
print(options[:5])

print("\nFirst few correct answers:")
print(correct_answers[:5])

# Prepare the data for model evaluation
# Format the options into a list of choices for each question
formatted_questions = []
for i in range(len(questions)):
    question = questions[i]
    options_list = options[i]
    correct_answer = correct_answers[i]
    
    # Split the options into a list of choices
    choices = []
    for key, value in options_list.items():
        choices.append(value)
    
    # Find the index of the correct answer in the choices
    correct_answer_index = choices.index(correct_answer)
    
    # Format the question with the options and the correct answer index
    formatted_question = {
        "question": question,
        "options": choices,
        "correct_answer": correct_answer_index
    }
    formatted_questions.append(formatted_question)

# Display the first few formatted questions
print("\nFirst few formatted questions:")
print(formatted_questions[:5])