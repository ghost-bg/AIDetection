import pandas as pd
import random
import json
import uuid

# Load the cleaned dataset
df = pd.read_csv("data/output/careerguidance_cleaned.csv")

# Randomly sample half of the dataset
sampled_df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
# Format batch requests for ChatGPT API
batch_requests = []

for idx, row in sampled_df.iterrows():
    custom_id = f"request-{uuid.uuid4().hex[:8]}"
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a reddit user on the subreddit r/careersadvice, A place to discuss career options, to ask questions and give advice. Write a post with the following title: "
                },
                {
                    "role": "user",
                    "content": row["Title"]
                }
            ],
            "max_tokens": 1000
        }
    }
    batch_requests.append(request)

# Save to JSON file
jsonl_path = "data/input/reddit_batch_requests.jsonl"
with open(jsonl_path, "w") as f:
    for entry in batch_requests:
        json.dump(entry, f)
        f.write("\n")
