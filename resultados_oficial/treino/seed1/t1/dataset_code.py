from datasets import Dataset

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