import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index and documents
index = faiss.read_index("data/index/travel_index.faiss")
with open("data/index/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask user for input
query = input("Enter your travel query: ")

# Generate query embedding
query_embedding = model.encode([query]).astype("float32")

# Search in the index
k = 5  # Number of results
D, I = index.search(query_embedding, k)

# Print results
print("\nTop matching results:\n")
for i in I[0]:
    print("- ", documents[i])
    print()
