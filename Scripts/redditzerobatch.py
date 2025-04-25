#!/usr/bin/env python3

import pandas as pd
import json
from pathlib import Path

# Config
csv_path = "data/output/merged_reddit_human_ai.csv"
output_dir = Path("data/zeroshot_reddit_batch_requests")
output_dir.mkdir(parents=True, exist_ok=True)

model = "gpt-4o-mini-2024-07-18"

# Load CSV
df = pd.read_csv(csv_path)
total_rows = len(df)

# Chunking rules
first_chunk_size = 5_000
other_chunk_size = 5_000

def create_request(row, idx):
    system_msg = {
        "role": "system",
        "content": "You are a linguistic analyst evaluating whether a social media post was written by a human or AI. Answer with only 'Human' or 'AI'."
    }
    user_msg = {
        "role": "user",
        "content": f"Title: {row['Title']}\Body: {row['Body']}\n\nIs this response written by a human or by an AI?"
    }

    return {
        "custom_id": f"request-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",  # or gpt-4-turbo / gpt-3.5-turbo
            "messages": [system_msg, user_msg],
            "temperature": 0,
            "max_tokens": 10
        }
    }

# Split and write chunks
chunk_start = 0
chunk_id = 0

while chunk_start < total_rows:
    if chunk_id == 0:
        chunk_size = first_chunk_size
    else:
        chunk_size = other_chunk_size

    chunk_end = min(chunk_start + chunk_size, total_rows)
    chunk_df = df.iloc[chunk_start:chunk_end]

    output_path = output_dir / f"requests_chunk_{chunk_id:02}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in chunk_df.iterrows():
            request = create_request(row, idx + chunk_start)  # preserve global index
            f.write(json.dumps(request) + "\n")

    print(f"Saved chunk {chunk_id} ({chunk_end - chunk_start} rows) to {output_path}")
    chunk_start += chunk_size
    chunk_id += 1
