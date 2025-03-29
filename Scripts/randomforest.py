#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('data/output/updated_combined_dataset_no_duplicates.csv')
df = df.dropna(subset=['answer'])

def extract_features(text):
    text = str(text)  # Convert to string in case it's NaN or another type
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

features = df['answer'].apply(extract_features)
features_df = pd.DataFrame(features.tolist())
features_df = features_df.dropna()

features_df['label'] = df['label']

filtered_df = features_df[
    (features_df['char_count'] < 10000) &
    (features_df['word_count'] < 2000)
]


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = filtered_df.drop('label', axis=1)
y = filtered_df['label']

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
y = y.loc[X.index]

y = y.dropna()
X = X.loc[y.index]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind='barh', title='Feature Importance')
plt.tight_layout()
plt.show()

joblib.dump(clf, 'models/Models/random_forest_ai_detector.joblib')
