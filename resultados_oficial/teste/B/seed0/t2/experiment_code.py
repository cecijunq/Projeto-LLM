import pandas as pd
from datasets import load_dataset
import os
from openai import OpenAI

# Load the SST-2 dataset from HuggingFace
dataset = load_dataset('stanfordnlp/sst2', split='test[:200]')

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Display the first few rows to verify the data
print("First few rows of the dataset:")
print(df.head())

# Save the dataset to a CSV file for later use
df.to_csv('sst2_subset.csv', index=False)

# Initialize OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Define function to evaluate model responses
def evaluate_model(prompt):
    r = client.chat.completions.create(
        model=os.environ['AGENTLAB_LLM_MODEL'],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0
    )
    answer = r.choices[0].message.content.strip()
    return answer

# Analyze model performance on sentences with and without negation
negation_sentences = df[df['sentence'].str.contains('not|never|no|none|neither|nothing|nobody|nowhere|noone|cannot|can not|should not|will not|would not|might not|could not|must not|had not|has not|have not|did not|does not|do not|is not|are not|was not|were not|has not|had not|will not|would not|could not|should not|may not|might not|must not|can not|cannot|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hadn\'t|will\'t|would\'t|could\'t|should\'t|may\'t|might\'t|must\'t|can\'t')]['sentence']
non_negation_sentences = df[~df['sentence'].str.contains('not|never|no|none|neither|nothing|nobody|nowhere|noone|cannot|can not|should not|will not|would not|might not|could not|must not|had not|has not|have not|did not|does not|do not|is not|are not|was not|were not|has not|had not|will not|would not|could not|should not|may not|might not|must not|can not|cannot|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hadn\'t|will\'t|would\'t|could\'t|should\'t|may\'t|might\'t|must\'t|can\'t')]['sentence']

# Evaluate model on negation and non-negation sentences
print("\nModel performance on sentences with negation words:")
for sentence in negation_sentences[:50]:
    prompt = f"Classify the sentiment of the following sentence: '{sentence}' (Possible answers: positive, negative)"
    answer = evaluate_model(prompt)
    print(f"Sentence: {sentence}")
    print(f"Model Response: {answer}")
    print("-" * 50)

print("\nModel performance on sentences without negation words:")
for sentence in non_negation_sentences[:50]:
    prompt = f"Classify the sentiment of the following sentence: '{sentence}' (Possible answers: positive, negative)"
    answer = evaluate_model(prompt)
    print(f"Sentence: {sentence}")
    print(f"Model Response: {answer}")
    print("-" * 50)