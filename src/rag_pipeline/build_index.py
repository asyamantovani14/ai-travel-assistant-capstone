import os
import pickle
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def build_index():
    print("Loading dataset...")
    dataset_dict = load_dataset("osunlp/TravelPlanner", "train")
    train_data = dataset_dict["train"]

    print("Extracting and combining fields...")
    documents = []
    for item in train_data:
        query = item.get("query", "")
        references = item.get("reference_information", "")
        references_text = "\n".join(str(r) for r in references) if isinstance(references, list) else str(references)
        documents.append(query + "\n" + references_text)

    print("Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("data/index", exist_ok=True)
    faiss.write_index(index, "data/index/travel_index.faiss")
    with open("data/index/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("Index built and saved with query + reference_information.")

if __name__ == "__main__":
    build_index()
