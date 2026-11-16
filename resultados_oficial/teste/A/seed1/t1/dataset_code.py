import json
from datasets import Dataset

# Define the examples
examples = [
    {"word": "see", "context1": "I see what you mean.", "context2": "I see what you mean.", "label": "Yes"},
    {"word": "see", "context1": "He saw the movie.", "context2": "He sees the movie.", "label": "No"},
    {"word": "make", "context1": "She made a cake.", "context2": "She makes a cake.", "label": "No"},
    {"word": "make", "context1": "He made a mistake.", "context2": "He makes a mistake.", "label": "No"},
    {"word": "think", "context1": "I think you're right.", "context2": "I think you're right.", "label": "Yes"},
    {"word": "think", "context1": "He thought about it.", "context2": "He thinks about it.", "label": "No"},
    {"word": "write", "context1": "She wrote a letter.", "context2": "She writes a letter.", "label": "No"},
    {"word": "run", "context1": "He ran quickly.", "context2": "He runs quickly.", "label": "No"},
    {"word": "break", "context1": "The glass broke.", "context2": "The glass breaks.", "label": "No"},
    {"word": "have", "context1": "I have a car.", "context2": "I have a car.", "label": "Yes"},
    {"word": "have", "context1": "I have a car.", "context2": "I has a car.", "label": "No"},
    {"word": "tell", "context1": "She told me a story.", "context2": "She tells me a story.", "label": "No"},
    {"word": "play", "context1": "They played football.", "context2": "They plays football.", "label": "No"},
    {"word": "go", "context1": "He went to the store.", "context2": "He goes to the store.", "label": "No"},
    {"word": "look", "context1": "She looked at the picture.", "context2": "She looks at the picture.", "label": "No"},
    {"word": "call", "context1": "He called his friend.", "context2": "He calls his friend.", "label": "No"},
    {"word": "take", "context1": "She took the book.", "context2": "She takes the book.", "label": "No"},
    {"word": "do", "context1": "I do my homework.", "context2": "I does my homework.", "label": "No"},
]

# Create the dataset
dataset = Dataset.from_dict({
    "word": [example["word"] for example in examples],
    "context1": [example["context1"] for example in examples],
    "context2": [example["context2"] for example in examples],
    "label": [example["label"] for example in examples],
})

# Display the dataset
print(dataset)