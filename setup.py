#!/usr/bin/env python3
from datasets import load_dataset
from openai import OpenAI
import yaml
import pandas as pd
import json
from tqdm import tqdm



ds = load_dataset("toughdata/quora-question-answer-dataset", split="train")

print(ds[0])
print(ds[1])

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['openai_api_key']
client = OpenAI(api_key=api_key)

def get_chatgpt_response(question):
    try:
        # Using the new API interface
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a human user answering a question on the website Quora."},
                {"role": "user", "content": question}],
            temperature = 0.7
        )
        return response.choices[0].message
    except Exception as e:
        print(f"Error: {e}")
        return None
print(get_chatgpt_response("hello"))
