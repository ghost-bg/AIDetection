#!/usr/bin/env python3

import pandas as pd
import json

# File paths
csv_file = "combined_dataset.csv"  # Existing CSV file
jsonl_file = "cleaned_batch.jsonl"  # JSONL file to append
output_file = "updated_combined_dataset.csv"  # Output file

# Load the existing CSV file
csv_data = pd.read_csv(csv_file)

# Load the JSONL file and add the label
jsonl_data = []
with open(jsonl_file, "r") as infile:
    for line in infile:
        record = json.loads(line)
        record["label"] = 1  # Add label to each record
        jsonl_data.append(record)

# Convert JSONL data to a DataFrame
jsonl_df = pd.DataFrame(jsonl_data)

# Combine the two datasets
combined_data = pd.concat([csv_data, jsonl_df], ignore_index=True)

# Save the updated dataset to a new CSV file
combined_data.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}.")
