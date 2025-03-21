#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np

# File paths
chatgpt_file = "cleaned_batch.jsonl"
quad_file = "Quora-QuAD.jsonl"

# Load ChatGPT responses
chatgpt_data = []
with open(chatgpt_file, "r") as f:
    for line in f:
        record = json.loads(line)
        chatgpt_data.append({"question": record["question"], "answer": record["answer"], "label": 1})

# Load human responses
quad_data = []
with open(quad_file, "r") as f:
    for line in f:
        record = json.loads(line)
        quad_data.append({"question": record["question"], "answer": record["answer"], "label": 0})

# Combine datasets
combined_data = chatgpt_data + quad_data

np.random.shuffle(combined_data)
df = pd.DataFrame(combined_data)

# Text cleaning (example: normalize whitespace)
df["answer"] = df["answer"].str.replace(r"\s+", " ", regex=True).str.strip()

# Save combined dataset to a CSV file
df.to_csv("combined_dataset.csv", index=False)
print("Dataset saved as combined_dataset.csv")
