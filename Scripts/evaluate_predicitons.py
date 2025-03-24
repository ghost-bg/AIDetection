import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/output/zero_shot_results.csv", encoding="ISO-8859-1", low_memory=False)

# Drop rows with missing or blank predictions
df = df[df["chatgpt_prediction"].notna()]
df = df[df["chatgpt_prediction"].astype(str).str.strip() != ""]

# Normalize prediction
def map_prediction(pred):
    pred = str(pred).strip().lower()
    if pred == "human":
        return 0
    elif pred == "ai":
        return 1
    return None

df["predicted_label"] = df["chatgpt_prediction"].apply(map_prediction)

# Drop invalid predictions
df = df[df["predicted_label"].notna()]

df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").astype("Int64")


df = df.dropna(subset=["label", "predicted_label"])

y_true = df["label"].astype(int)
y_pred = df["predicted_label"].astype(int)

print(f" Evaluating {len(df)} rows.")

# Report
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["0", "1"]))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
