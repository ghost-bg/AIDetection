#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path

# File paths
results_path = "data/completed_zero_shot/1.jsonl"
output_csv = "data/output/parsed_results.csv"

# Container for parsed outputs
parsed_rows = []

# Parse each line
with open(results_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)

        custom_id = entry.get("custom_id")
        response = entry.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if choices:
            answer = choices[0]["message"]["content"].strip()
        else:
            answer = "NO_RESPONSE"

        parsed_rows.append({
            "custom_id": custom_id,
            "chatgpt_prediction": answer
        })

# Convert to DataFrame and save
df = pd.DataFrame(parsed_rows)
df.to_csv(output_csv, index=False)
print(f"✅ Saved parsed results to {output_csv}")

# Add index column to original data
original_df = pd.read_csv("data/output/updated_combined_dataset_no_duplicates.csv")
original_df["custom_id"] = ["request-" + str(i) for i in range(len(original_df))]

# Load predictions
pred_df = pd.read_csv("data/output/parsed_results.csv")

# Merge
merged_df = pd.merge(original_df, pred_df, on="custom_id", how="left")
merged_df["is_correct"] = merged_df["label"].str.lower() == merged_df["chatgpt_prediction"].str.lower()

# Save merged results
merged_df.to_csv("data/output/final_evaluation.csv", index=False)
print("✅ Merged predictions with original labels and saved evaluation file.")
