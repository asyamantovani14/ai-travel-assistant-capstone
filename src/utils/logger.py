import os
from datetime import datetime

def log_interaction(query, matched_docs, response, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "query_log.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"--- Query at {timestamp} ---\n")
        f.write(f"User Query:\n{query}\n\n")
        f.write("Top Matching Documents:\n")
        for i, doc in enumerate(matched_docs, 1):
            f.write(f"{i}. {doc}\n")
        f.write("\nGenerated Response:\n")
        f.write(f"{response}\n")
        f.write("=" * 80 + "\n\n")
