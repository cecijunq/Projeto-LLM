from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset('AlgorithmicResearchGroup/math_reasoning_automated_problem_solving_with_code_track_3', 'test', split='train[:200]')

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Display the first few rows to verify the data
print("First few rows of the dataset:")
print(df.head())

# Save the dataset to a CSV file for further use (optional)
df.to_csv('math_questions.csv', index=False)