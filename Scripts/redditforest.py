#!/usr/bin/env python3

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# Load the new test set
new_df = pd.read_csv('data/output/merged_reddit_human_ai.csv')
new_df = new_df.dropna(subset=['Body'])

# Load the trained model and TF-IDF vectorizer
clf = joblib.load('models/Models/random_forest_ai_detector.joblib')
vectorizer = joblib.load('models/Models/tfidf_vectorizer.joblib')

# Define the feature extraction function (same as training)
def extract_features(text):
    text = str(text)
    words = text.split()
    word_lengths = [len(word) for word in words] if words else [0]
    return {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': sum(word_lengths) / len(word_lengths),
        'sentence_count': text.count('.') + text.count('!') + text.count('?'),
        'punctuation_ratio': sum(1 for c in text if c in '.,;!?') / len(text) if len(text) > 0 else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
    }

# Extract handcrafted features
new_features = new_df['Body'].apply(extract_features)
new_features_df = pd.DataFrame(new_features.tolist())
new_features_df = new_features_df.replace([np.inf, -np.inf], np.nan).dropna()

# Align labels with the cleaned features
y_new = new_df.loc[new_features_df.index, 'Label']

# Generate TF-IDF features on 'Body'
X_tfidf_new = vectorizer.transform(new_df.loc[new_features_df.index, 'Body'])

# Combine TF-IDF and handcrafted features
X_combined_new = hstack([X_tfidf_new, new_features_df.values])

# Predict
y_new_pred = clf.predict(X_combined_new)

# Evaluation
print("\nClassification Report on New Test Set:\n")
print(classification_report(y_new, y_new_pred))

# Confusion matrix
cm = confusion_matrix(y_new, y_new_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)


print("\nConfusion Matrix:\n", cm)

# Optional: display the matrix
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on New Test Set")
plt.show()
