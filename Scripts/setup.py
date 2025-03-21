#!/usr/bin/env python3
from datasets import load_dataset
from openai import OpenAI
import yaml
import json
from tqdm import tqdm
import os

# Load the dataset

ds = load_dataset("toughdata/quora-question-answer-dataset", split="train")

# Load API key from config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['openai_api_key']
client = OpenAI(api_key=api_key)

def get_chatgpt_response(question):
    """Query ChatGPT for a response to the given question."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a human user answering a question on the website Quora."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred during response generation."

# Output file
output_file = "data/output/quora_chatgpt_answers.jsonl"

# Load already processed questions
processed_questions = set()
if os.path.exists(output_file):
    with open(output_file, 'r') as file:
        print("Loading already processed questions...")
        for line in tqdm(file, desc="Loading progress"):
            try:
                entry = json.loads(line.strip())
                processed_questions.add(entry['question'])
            except json.JSONDecodeError:
                continue

# Process the dataset
print("Processing new questions...")
with open(output_file, 'a') as file:
    for i in tqdm(range(len(ds)), desc="Processing questions"):
        question = ds[i]['question']

        if question in processed_questions:
            continue

        answer = get_chatgpt_response(question)
        output_data = {"question": question, "answer": answer}

        file.write(json.dumps(output_data) + '\n')
        processed_questions.add(question)

print(f"Finished processing questions. Results saved to {output_file}.")
