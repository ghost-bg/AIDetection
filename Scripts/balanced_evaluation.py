#!/usr/bin/env python3

import pandas as pd

# Load the full evaluated dataset
df = pd.read_csv("data/output/zero_shot_results.csv", encoding = "ISO-8859-1", low_memory=False)

# Make sure predictions are valid and labels are numeric
df = df[df["chatgpt_prediction"].notna()]
df["predicted_label"] = df["chatgpt_prediction"].str.strip().str.lower().map({"human": 0, "ai": 1})
df = df[df["predicted_label"].notna()]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label", "predicted_label"])

# Separate into two classes
ai_rows = df[df["label"] == 1]
human_rows = df[df["label"] == 0]

# Sample an equal number of human rows to match AI count
human_sample = human_rows.sample(n=len(ai_rows), random_state=42)

# Combine into balanced set
balanced_df = pd.concat([ai_rows, human_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset created: {len(balanced_df)} rows ({len(ai_rows)} per class)")

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

y_true = balanced_df["label"].astype(int)
y_pred = balanced_df["predicted_label"].astype(int)

print("\nBalanced Classification Report:")
print(classification_report(y_true, y_pred, target_names=["0", "1"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Optionally save
balanced_df.to_csv("data/output/balanced_evaluation.csv", index=False)
