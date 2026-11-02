from datasets import Dataset
import os
from openai import OpenAI

# Define a small hand-written list of examples for differential diagnosis
examples = [
    {
        "question": "A patient presents with fever, cough, and shortness of breath. What are the possible diagnoses?",
        "answer": "Pneumonia, Tuberculosis, Bronchitis, Asthma"
    },
    {
        "question": "A patient has abdominal pain, nausea, and vomiting. What are the possible diagnoses?",
        "answer": "Gastritis, Gastroenteritis, Pancreatitis, Appendicitis"
    },
    {
        "question": "A patient presents with chest pain, shortness of breath, and sweating. What are the possible diagnoses?",
        "answer": "Myocardial Infarction, Pneumothorax, Pericarditis, Pulmonary Embolism"
    },
    {
        "question": "A patient has headache, fatigue, and weight loss. What are the possible diagnoses?",
        "answer": "Tuberculosis, Hypothyroidism, Lymphoma, Anemia"
    },
    {
        "question": "A patient presents with rash, itching, and swelling. What are the possible diagnoses?",
        "answer": "Allergic Reaction, Hives, Contact Dermatitis, Urticaria"
    },
    {
        "question": "A patient presents with jaundice, dark urine, and pale stools. What are the possible diagnoses?",
        "answer": "Hepatitis, Gallstones, Liver Cancer, Bile Duct Obstruction"
    },
    {
        "question": "A patient has sudden onset of severe headache and neck stiffness. What are the possible diagnoses?",
        "answer": "Meningitis, Subarachnoid Hemorrhage, Encephalitis, Brain Tumor"
    },
    {
        "question": "A patient presents with rash, fever, and joint pain. What are the possible diagnoses?",
        "answer": "Rheumatic Fever, Lyme Disease, Dengue Fever, Scarlet Fever"
    },
    {
        "question": "A patient presents with palpitations, shortness of breath, and chest discomfort. What are the possible diagnoses?",
        "answer": "Atrial Fibrillation, Hypertension, Myocardial Infarction, Pericarditis"
    },
    {
        "question": "A patient has chronic diarrhea, weight loss, and abdominal pain. What are the possible diagnoses?",
        "answer": "Irritable Bowel Syndrome, Inflammatory Bowel Disease, Lactose Intolerance, Celiac Disease"
    }
]

# Convert the list of examples into a Dataset
dataset = Dataset.from_list(examples)

# Display the first few examples
print("First few examples:")
print(dataset[:3])

# Initialize the OpenAI client
client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))

# Function to evaluate the model
def evaluate_model(question, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=256,
        temperature=0
    )
    return response.choices[0].message.content

# Define prompts
base_prompt = "Please provide the possible diagnoses for the following medical question."
differential_prompt = "Please provide the possible diagnoses for the following medical question, making sure to list all potential differential diagnoses."

# Evaluate the model with and without the differential diagnosis prompt
print("\nResults without differential diagnosis prompt:")
for example in dataset:
    question = example["question"]
    answer = example["answer"]
    model_response = evaluate_model(question, os.environ['AGENTLAB_LLM_MODEL'])
    print(f"Question: {question}")
    print(f"Model Response: {model_response}")
    print(f"Correct Answer: {answer}")
    print("-" * 80)

print("\nResults with differential diagnosis prompt:")
for example in dataset:
    question = example["question"]
    answer = example["answer"]
    model_response = evaluate_model(differential_prompt + " " + question, os.environ['AGENTLAB_LLM_MODEL'])
    print(f"Question: {question}")
    print(f"Model Response: {model_response}")
    print(f"Correct Answer: {answer}")
    print("-" * 80)