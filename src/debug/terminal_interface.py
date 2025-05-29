import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.rag_pipeline.generate_response import generate_response


def main():
    # Load FAISS index and docs
    index = faiss.read_index("data/index/travel_index.faiss")
    with open("data/index/documents.pkl", "rb") as f:
        all_documents = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Filters ---
    print("\n--- Optional Filters ---")
    def ask_boolean(prompt): return input(prompt).strip().lower() == "y"

    allowed_countries, activity_keywords = [], []
    min_days = max_days = budget_limit = None

    if ask_boolean("Filter by country? (y/n): "):
        allowed_countries = input("Countries (comma separated): ").lower().split(',')

    if ask_boolean("Filter by trip duration? (y/n): "):
        min_days = int(input("Min days: "))
        max_days = int(input("Max days: "))

    if ask_boolean("Filter by preferred activities? (y/n): "):
        activity_keywords = input("Activities (comma separated): ").lower().split(',')

    if ask_boolean("Set a max budget? (y/n): "):
        budget_limit = float(input("Max budget in USD: "))

    # Apply filters
    filtered_documents = []
    for doc in all_documents:
        doc_lower = doc.lower()
        include = True

        if allowed_countries:
            include &= any(c.strip() in doc_lower for c in allowed_countries)

        if min_days is not None and max_days is not None:
            match = re.search(r"(\d+)\s*[- ]?day[s]?", doc_lower)
            if match:
                days = int(match.group(1))
                include &= min_days <= days <= max_days
            else:
                include = False

        if activity_keywords:
            include &= any(a.strip() in doc_lower for a in activity_keywords)

        if budget_limit is not None:
            match = re.search(r"\$?(\d{3,5})", doc)
            if match:
                include &= float(match.group(1)) <= budget_limit
            else:
                include = False

        if include:
            filtered_documents.append(doc)

    if not filtered_documents:
        print("No documents matched the filters.")
        return

    query = input("\nEnter your travel query: ")
    query_embedding = model.encode([query]).astype("float32")

    print("Ranking filtered documents...")
    filtered_embeddings = model.encode(filtered_documents, show_progress_bar=True).astype("float32")
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)
    k = min(5, len(filtered_documents))
    distances, indices = temp_index.search(query_embedding, k)

    matched_docs = [filtered_documents[idx] for idx in indices[0]]
    for i, doc in enumerate(matched_docs):
        score = 1 / (1 + distances[0][i])
        print(f"\nMatch {i+1} (score: {score:.4f}):\n{doc}")

    print("\n--- Travel Assistant Response ---\n")
    print(generate_response(query, matched_docs))

if __name__ == "__main__":
    main()
