import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

df_path = "data/input/combined_dataset_no_duplicates.csv"

df = pd.read_csv(df_path)

df["answer"] = df["answer"].astype(str)

X = df["answer"]
y = df["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Verify independence of training and testing sets
train_answers = set(X_train)  # Convert training answers to a set
test_answers = set(X_test)  # Convert testing answers to a set

# Find the intersection of training and testing answers
overlap = train_answers.intersection(test_answers)

# Report results
if overlap:
    print(f"Found {len(overlap)} overlapping answers between training and testing sets.")
    print("Sample overlaps:", list(overlap)[:5])  # Display a few overlapping answers
else:
    print("No overlapping answers between training and testing sets.")
# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")

tqdm.pandas(desc="Transforming TF-IDF")
X_train_tfidf = tfidf.fit_transform(tqdm(X_train, desc="Fitting TF-IDF"))
X_test_tfidf = tfidf.transform(tqdm(X_test, desc="Transforming TF-IDF"))

model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
y_train_pred = model.predict(X_train_tfidf)
# Test performance (already in your code)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(tfidf, "models/Vectoriser/tfidf_vectorizer.pkl")
joblib.dump(model, "models/Models/logistic_regression_model.pkl")
print("Model and vectorizer saved!")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", cm)

# Optional: display the confusion matrix as a plot
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')  # You can change the colormap if you'd like
plt.title("Confusion Matrix")
plt.show()
