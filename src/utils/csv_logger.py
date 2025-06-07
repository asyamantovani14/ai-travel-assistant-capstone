import os
import csv
import json
from datetime import datetime

CSV_PATH = "logs/responses.csv"

def save_response_to_csv(query, response, model, entities, prompt, log_file="logs/responses.csv"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    headers = ["timestamp", "query", "model", "entities", "prompt", "response"]
    write_header = not os.path.exists(log_file)

    with open(log_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "model": model,
            "entities": json.dumps(entities, ensure_ascii=False),
            "prompt": prompt,
            "response": response
        })

