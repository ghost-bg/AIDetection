#!/usr/bin/env python3

import pandas as pd
import json

# Load the original dataset
df = pd.read_csv('data/output/merged_reddit_human_ai.csv')
df = df.dropna(subset=['Body', 'Label'])

# Load the GPT predictions
with open('data/output/zeroshotredditresults.jsonl', 'r') as f:  # Replace with actual path
    responses = [json.loads(line) for line in f]

# Build a dataframe from the GPT predictions
gpt_preds = []
for entry in responses:
    idx = int(entry['custom_id'].split('-')[1])  # Extract index from "request-<idx>"
    prediction = entry['response']['body']['choices'][0]['message']['content'].strip()
    gpt_preds.append((idx, prediction))

preds_df = pd.DataFrame(gpt_preds, columns=['index', 'gpt_prediction'])
preds_df['gpt_prediction'] = preds_df['gpt_prediction'].map({'Human': 0, 'AI': 1})  # Match label format

# Add the ground truth
df = df.reset_index(drop=True)
truth = df[['Label']].copy()
truth['index'] = truth.index

# Merge
merged = pd.merge(truth, preds_df, on='index')

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = merged['Label']
y_pred = merged['gpt_prediction']

print("Zero-Shot Classification Report:\n")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'AI'])
print(cm)
disp.plot(cmap='Blues')
plt.title("Zero-Shot Confusion Matrix (GPT-4o-mini)")
plt.show()
