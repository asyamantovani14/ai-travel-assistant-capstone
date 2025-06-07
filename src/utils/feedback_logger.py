import os
import csv
from datetime import datetime

def save_feedback(query, rag_response, llm_response, selected, notes=None, log_path="logs/feedback.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    is_new = not os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "timestamp", "query", "preferred", "notes", "rag_response", "llm_response"
        ])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "preferred": selected,
            "notes": notes or "",
            "rag_response": rag_response,
            "llm_response": llm_response
        })
