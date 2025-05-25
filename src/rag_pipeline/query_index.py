import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from generate_response import generate_response

# Load FAISS index and documents
index = faiss.read_index("data/index/travel_index.faiss")
with open("data/index/documents.pkl", "rb") as f:
    all_documents = pickle.load(f)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask user for filters
print("\n--- Optional Filters ---")

# Countries
apply_country_filter = input("Do you want to filter by country? (y/n): ").strip().lower() == "y"
allowed_countries = []
if apply_country_filter:
    countries_input = input("Enter country names separated by commas (e.g. Italy, Spain, France): ")
    allowed_countries = [c.strip().lower() for c in countries_input.split(",")]

# Duration
apply_duration_filter = input("Do you want to filter by trip duration? (y/n): ").strip().lower() == "y"
min_days, max_days = None, None
if apply_duration_filter:
    min_days = int(input("Minimum number of days: "))
    max_days = int(input("Maximum number of days: "))

# Activities
apply_activity_filter = input("Do you want to filter by preferred activities? (y/n): ").strip().lower() == "y"
activity_keywords = []
if apply_activity_filter:
    activity_input = input("Enter keywords for activities (e.g. beach, hiking, museum): ")
    activity_keywords = [a.strip().lower() for a in activity_input.split(",")]

# Budget
apply_budget_filter = input("Set a max budget? (y/n): ").strip().lower() == "y"
budget_limit = None
if apply_budget_filter:
    budget_limit = float(input("Enter max budget in USD: "))

filtered_documents = []
filtered_indices = []

import re

for i, doc in enumerate(all_documents):
    doc_lower = doc.lower()
    include = True

    if apply_country_filter:
        include &= any(country in doc_lower for country in allowed_countries)

    if apply_duration_filter:
        match = re.search(r"(\d+)\s*[- ]?day[s]?", doc_lower)
        if match:
            days = int(match.group(1))
            include &= min_days <= days <= max_days
        else:
            include = False

    if apply_activity_filter:
        include &= any(activity in doc_lower for activity in activity_keywords)

    if apply_budget_filter:
        match = re.search(r"\$?(\d{3,5})", doc)
        if match:
            include &= float(match.group(1)) <= budget_limit
        else:
            include = False

    if include:
        filtered_documents.append(doc)
        filtered_indices.append(i)

# If no match, abort early
if not filtered_documents:
    print(" No documents matched your filters. Try relaxing them and try again.")
    exit()

# Encode query
query = input("\nEnter your travel query: ")
query_embedding = model.encode([query]).astype("float32")

# Create FAISS sub-index
print("\nRanking filtered documents...")
filtered_embeddings = model.encode(filtered_documents, show_progress_bar=True).astype("float32")
temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
temp_index.add(filtered_embeddings)
k = min(5, len(filtered_documents))
distances, indices = temp_index.search(query_embedding, k)

# Print results
print("\nTop matching results (with similarity scores):\n")
matched_docs = []
for i, idx in enumerate(indices[0]):
    similarity = 1 / (1 + distances[0][i])
    print(f"- {filtered_documents[idx]} (score: {similarity:.4f})\n")
    matched_docs.append(filtered_documents[idx])

# Generate final response
response = generate_response(query, matched_docs)
print("\n--- Travel Assistant Response ---\n")
print(response)
