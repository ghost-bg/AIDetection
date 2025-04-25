#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv('data/output/updated_combined_dataset_no_duplicates.csv')
df = df.dropna(subset=['answer'])

# Handcrafted feature extraction
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

# Apply feature extraction
features_df = df['answer'].apply(extract_features).apply(pd.Series)
features_df['label'] = df['label']

# Filter data
features_df = features_df.dropna()
features_df = features_df[(features_df['char_count'] < 10000) & (features_df['word_count'] < 2000)]

# Separate features and label
X_handcrafted = features_df.drop('label', axis=1).replace([np.inf, -np.inf], np.nan).dropna()
y = features_df.loc[X_handcrafted.index, 'label']

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=300)
X_tfidf = vectorizer.fit_transform(df.loc[X_handcrafted.index, 'answer'])

# Combine handcrafted and TF-IDF features
X_combined = hstack([X_tfidf, X_handcrafted.values])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Grouped feature importance: TF-IDF vs handcrafted
n_tfidf = len(vectorizer.get_feature_names_out())
n_handcrafted = X_handcrafted.shape[1]

tfidf_importances = clf.feature_importances_[:n_tfidf]
handcrafted_importances = clf.feature_importances_[n_tfidf:]

# Combine TF-IDF importance into one group
grouped_importances = pd.Series({'TF-IDF (combined)': tfidf_importances.sum()})

# Add individual handcrafted feature importances
for name, importance in zip(X_handcrafted.columns, handcrafted_importances):
    grouped_importances[name] = importance

# Plot grouped importances
grouped_importances.sort_values(ascending=True).plot(kind='barh', title='Grouped Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Save model and vectorizer
joblib.dump(clf, 'models/Models/random_forest_ai_detector.joblib')
joblib.dump(vectorizer, 'models/Models/tfidf_vectorizer.joblib')
