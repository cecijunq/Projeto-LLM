from datasets import load_dataset
import re

# Load the GSM8K dataset with the specified split
dataset = load_dataset('openai/gsm8k', 'main', split='test[:200]')

# Function to reverse the word order of a question while preserving mathematical content
def reverse_question(text):
    # Split the text into parts separated by <<...>>
    parts = re.split(r'<<[^>]*?>>', text)
    reversed_parts = []
    for part in parts:
        # Split the part into words and reverse the order
        words = part.split()
        reversed_words = words[::-1]
        reversed_part = ' '.join(reversed_words)
        reversed_parts.append(reversed_part)
    # Join the parts back together
    reversed_text = '<<'.join(reversed_parts) + '>>'
    return reversed_text

# Apply the function to reverse the word order of each question
dataset = dataset.map(lambda example: {
    'original_question': example['question'],
    'reversed_question': reverse_question(example['question']),
    'answer': example['answer']
})

# Display the first few examples to confirm the structure
print(dataset[:5])