#!/usr/bin/env python3

import pandas as pd

# Load the dataset
df = pd.read_csv("data/input/careerguidance_data.csv")

# Step 1: Remove rows where 'Body' is null
df = df[df['Body'].notnull()]

# Step 2: Keep only the 'Title' and 'Body' columns
df_cleaned = df[['Title', 'Body']]

# Optional: Save the cleaned dataset to a new file
df_cleaned.to_csv("careerguidance_cleaned.csv", index=False)

print("Cleaning complete. Cleaned dataset saved as 'careerguidance_cleaned.csv'")
