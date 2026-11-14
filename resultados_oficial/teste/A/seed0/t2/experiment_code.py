import os
import pandas as pd
from datasets import load_dataset
from openai import OpenAI

# Set up the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Load the SST-2 dataset with a small subset
sst2 = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Define negation words
negation_words = {'not', 'never', 'no', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}

# Split the dataset into two subsets: with and without negation words
with_negation = []
without_negation = []

for item in sst2:
    sentence = item['sentence']
    label = item['label']
    has_negation = any(word in sentence.lower() for word in negation_words)
    
    if has_negation:
        with_negation.append({"sentence": sentence, "label": label})
    else:
        without_negation.append({"sentence": sentence, "label": label})

# Convert subsets to pandas DataFrames for easier handling
with_negation_df = pd.DataFrame(with_negation)
without_negation_df = pd.DataFrame(without_negation)

# Display the first few items from each subset
print("Examples with negation words:")
print(with_negation_df.head())
print("\nExamples without negation words:")
print(without_negation_df.head())

# Function to predict sentiment using the LLM
def predict_sentiment(text):
    prompt = f"Classify the sentiment of the following sentence as positive (1) or negative (0): {text}"
    response = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return int(answer) if answer in {'0', '1'} else 0  # Default to 0 if the response is unexpected

# Evaluate performance on subsets
def evaluate_subset(df):
    correct = 0
    total = len(df)
    for _, row in df.iterrows():
        predicted_label = predict_sentiment(row['sentence'])
        if predicted_label == row['label']:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Evaluate both subsets
accuracy_with_negation = evaluate_subset(with_negation_df)
accuracy_without_negation = evaluate_subset(without_negation_df)

# Print results
print("\nSentiment prediction accuracy on sentences with negation words:")
print(accuracy_with_negation)
print("\nSentiment prediction accuracy on sentences without negation words:")
print(accuracy_without_negation)