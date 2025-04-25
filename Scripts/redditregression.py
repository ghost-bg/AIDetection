#!/usr/bin/env python3

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the new test dataset
new_df = pd.read_csv('data/output/merged_reddit_human_ai.csv')

# Ensure the text column is string type
new_df['Body'] = new_df['Body'].astype(str)

# Load pre-trained TF-IDF vectorizer and logistic regression model
tfidf = joblib.load("models/Vectoriser/tfidf_vectorizer.pkl")
model = joblib.load("models/Models/logistic_regression_model.pkl")

# Transform the new dataset using the loaded vectorizer
X_new_tfidf = tfidf.transform(new_df['Body'])

# Extract labels (if available)
y_new = new_df['Label']

# Predict
y_new_pred = model.predict(X_new_tfidf)

# Evaluate
print("Classification Report on New Data:")
print(classification_report(y_new, y_new_pred))

# Confusion Matrix
cm = confusion_matrix(y_new, y_new_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on New Test Set")
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the confusion matrix
cm = confusion_matrix(y_new, y_new_pred)

# Print it in text form
print("Confusion Matrix:\n", cm)
