import os
import json
import random
import datetime
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define the prompts for anchoring bias
anchoring_prompts = []
for i in range(20):
    initial_number = random.randint(100, 1000)
    question = f"Estimate the population of a city that has an initial number of {initial_number} people."
    anchoring_prompts.append({"prompt": question, "type": "anchoring"})

# Define the prompts for confirmation bias
confirmation_prompts = []
for i in range(20):
    premise = f"Premise {i+1}: {random.choice(['The Earth is flat.', 'The Earth is round.'])}"
    question = f"Provide evidence supporting or contradicting the premise: {premise}"
    confirmation_prompts.append({"prompt": question, "type": "confirmation"})

# Define the neutral prompts
neutral_prompts = []
for i in range(20):
    question = f"What is the capital of {random.choice(['France', 'Germany', 'Italy', 'Spain', 'Japan', 'Brazil', 'Canada', 'Australia', 'India', 'China'])}?"
    neutral_prompts.append({"prompt": question, "type": "neutral"})

# Combine all prompts
all_prompts = anchoring_prompts + confirmation_prompts + neutral_prompts

# Function to send a prompt to the Ollama API and get the response
def send_prompt(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: " + str(response.status_code)

# Generate responses for all prompts
responses = []
for prompt in all_prompts:
    response = send_prompt(prompt["prompt"])
    responses.append({
        "prompt": prompt["prompt"],
        "response": response,
        "type": prompt["type"]
    })

# Save responses to a JSON file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"bias_analysis_results_{timestamp}.json"
with open(output_file, "w") as f:
    json.dump(responses, f, indent=4)

# Analyze the responses
analysis = {
    "timestamp": timestamp,
    "total_prompts": len(all_prompts),
    "anchoring_prompts": len(anchoring_prompts),
    "confirmation_prompts": len(confirmation_prompts),
    "neutral_prompts": len(neutral_prompts),
    "results": responses
}

# Save analysis to a JSON file
analysis_file = f"bias_analysis_summary_{timestamp}.json"
with open(analysis_file, "w") as f:
    json.dump(analysis, f, indent=4)

print(f"Responses saved to {output_file}")
print(f"Analysis saved to {analysis_file}")