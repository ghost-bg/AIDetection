#!/usr/bin/env python3

import pandas as pd

# Load the combined dataset
file_path = "updated_combined_dataset.csv"  # Replace with your file name

df = pd.read_csv(file_path)

# Remove duplicate answers, keeping the first occurrence
df_unique = df.drop_duplicates(subset="answer", keep="first")  # `keep="first"` retains the first occurrence

# Save the updated dataset to a new file
output_file = "updated_combined_dataset_no_duplicates.csv"
df_unique.to_csv(output_file, index=False)

# Print results
print(f"Original dataset had {len(df)} entries.")
print(f"Dataset after removing duplicates has {len(df_unique)} entries.")
print(f"Updated dataset saved to {output_file}.")
# Load the dataset
df = pd.read_csv("combined_dataset_no_duplicates.csv")

# Count the occurrences of each class
class_counts = df["label"].value_counts()

# Calculate the percentage of each class
class_distribution = class_counts / len(df) * 100

# Display results
print(class_counts)
print("Class Distribution:")
print(class_distribution)

# Visualize the distribution (optional)
import matplotlib.pyplot as plt

class_counts.plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class (0: Human, 1: AI)")
plt.ylabel("Number of Responses")
plt.show()
