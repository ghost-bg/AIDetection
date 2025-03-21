import json

# Input and output file paths
input_file = "Quora-QuAD.jsonl"  # Replace with your input file name
output_file = "trimmed_batch_requests.jsonl"  # Output file name

# Parameters for processing
start_index = 13233 # 0-based index, 3234th question is index 3233
max_requests = 20000

# Prepare batch requests
batch_requests = []
with open(input_file, "r") as infile:
    for index, line in enumerate(infile):
        if index < start_index:
            continue  # Skip lines before the start index
        if index >= start_index + max_requests:
            break  # Stop after processing the required number of requests

        data = json.loads(line)
        question = data["question"]

        # Prepare a single request
        request = {
            "custom_id": f"request-{index + 1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a human user answering a question on the website Quora."},
                    {"role": "user", "content": question}
                ],
                "max_tokens": 1000
            }
        }
        batch_requests.append(request)

# Write the trimmed batch requests to an output JSONL file
with open(output_file, "w") as outfile:
    for request in batch_requests:
        outfile.write(json.dumps(request) + "\n")

print(f"Trimmed batch requests saved to {output_file} with {len(batch_requests)} requests.")
