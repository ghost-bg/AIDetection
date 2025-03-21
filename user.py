import sys
import joblib

# Load the saved model and TF-IDF vectorizer
tfidf = joblib.load("models/Vectoriser/tfidf_vectorizer.pkl")
model = joblib.load("models/Models/logistic_regression_model.pkl")

print("Enter the response text (end input with an empty line):")

# Capture multi-line input
lines = []
while True:
    line = input()  # Reads a line of input
    if line.strip() == "":  # Stop when an empty line is entered
        break
    lines.append(line)

# Combine all lines into a single string
new_response = "\n".join(lines)

# Preprocess and classify the input
new_response_tfidf = tfidf.transform([new_response])
predicted_label = model.predict(new_response_tfidf)[0]

# Output the result
if predicted_label == 1:
    print("\nThe response is classified as AI-generated.")
else:
    print("\nThe response is classified as Human-generated.")
