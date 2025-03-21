#!/usr/bin/env python3

import json

# File paths
responses_file = "batchoutput2.jsonl"  # File with responses
questions_file = "trimmed_batch_requests.jsonl"  # File with questions
output_file = "cleaned_batch.jsonl"  # Output file

questions = []
with open(questions_file, "r") as qfile:
    for line in qfile:
        data = json.loads(line)
        # Extract the question content from nested structure
        question = data["body"]["messages"][1]["content"]
        questions.append(question)

# Process responses and align with questions
with open(responses_file, "r") as rfile, open(output_file, "w") as outfile:
    for idx, line in enumerate(rfile):
        response_data = json.loads(line)

        # Extract response content
        response = response_data["response"]["body"]["choices"][0]["message"]["content"]

        # Match the response to the corresponding question
        if idx < len(questions):
            question = questions[idx]
        else:
            question = "Unknown question"  # Fallback if there are more responses than questions

        # Write the question-response pair to the output file
        output_data = {
            "question": question,
            "answer": response
        }
        outfile.write(json.dumps(output_data) + "\n")

print(f"Question-response pairs saved to {output_file}")
