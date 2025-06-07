import os
import json
from datetime import datetime
import hashlib

def sanitize_filename(text):
    return "".join(c for c in text if c.isalnum() or c in (' ', '_', '-')).rstrip()

def log_interaction(
    query,
    matched_docs,
    response,
    extracted_entities=None,
    model=None,
    final_prompt=None,
    similarities=None,
    log_dir="logs"
):
    os.makedirs(log_dir, exist_ok=True)

    # Daily log file
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{date_str}_log.txt")

    # Optional: save also per-query log
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
    clean_query = sanitize_filename(query[:40]).replace(" ", "_")
    by_query_dir = os.path.join(log_dir, "by_query")
    os.makedirs(by_query_dir, exist_ok=True)
    query_log_file = os.path.join(by_query_dir, f"{clean_query}_{query_hash}.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_block = []

    log_block.append(f"--- Query at {timestamp} ---")
    log_block.append(f"Model Used: {model or 'unknown'}")
    log_block.append(f"User Query:\n{query}\n")

    log_block.append("Top Matching Documents:")
    for i, doc in enumerate(matched_docs, 1):
        doc_str = str(doc)  # cast anche se fosse una lista
        if similarities and i <= len(similarities):
            sim_score = similarities[i - 1]
            log_block.append(f"{i}. (score: {sim_score:.4f}) {doc_str}")
        else:
            log_block.append(f"{i}. {doc_str}")

    if similarities:
        sim_str = ", ".join([f"{s:.4f}" for s in similarities])
        log_block.append(f"\nAll Similarity Scores:\n{sim_str}")

    if extracted_entities:
        log_block.append("\nExtracted Entities:")
        log_block.append(str(json.dumps(extracted_entities, indent=2)))

    if final_prompt:
        log_block.append("\nFinal Prompt Sent to LLM:")
        log_block.append(str(final_prompt))

    log_block.append("\nGenerated Response:")
    log_block.append(str(response))
    log_block.append("=" * 100 + "\n")

    full_text = "\n".join(log_block)

    # Append to daily log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_text)

    # Save single query log too
    with open(query_log_file, "w", encoding="utf-8") as f:
        f.write(full_text)
