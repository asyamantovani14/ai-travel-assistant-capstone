import os
from datetime import datetime

def log_interaction(query, matched_docs, response, log_dir="logs"):
    """
    Log a user query, its matched documents, and the generated response to a file.

    Args:
        query (str): The user's query.
        matched_docs (list of str): List of top-matching documents.
        response (str): The generated response.
        log_dir (str): Directory where the log file will be saved.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "query_log.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"--- Query at {timestamp} ---\n")
        f.write(f"User Query:\n{str(query)}\n\n")

        f.write("Top Matching Documents:\n")
        for i, doc in enumerate(matched_docs, 1):
            f.write(f"{i}. {str(doc)}\n")

        f.write("\nGenerated Response:\n")
        f.write(f"{str(response)}\n")
        f.write("=" * 80 + "\n\n")
