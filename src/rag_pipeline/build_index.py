from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# Load dataset
print("Loading dataset...")
dataset_dict = load_dataset("osunlp/TravelPlanner", "train")
train_data = dataset_dict["train"]

# Print a sample
print("Sample entry:", train_data[0])
print("Columns:", train_data.column_names)

# Combine query + reference_information
print("Extracting and combining fields...")
documents = []
for item in train_data:
    query = item.get("query", "")
    references = item.get("reference_information", "")
    
    # In alcuni casi, reference_information può essere una stringa serializzata
    if isinstance(references, list):
        references_text = "\n".join(str(r) for r in references)
    else:
        references_text = str(references)
    
    full_text = query + "\n" + references_text
    documents.append(full_text)

# Generate embeddings
print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build index
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save
os.makedirs("data/index", exist_ok=True)
faiss.write_index(index, "data/index/travel_index.faiss")
with open("data/index/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Index built and saved with query + reference_information.")
