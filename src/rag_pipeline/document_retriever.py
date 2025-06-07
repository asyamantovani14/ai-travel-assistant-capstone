from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_faiss_index(documents, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, model

def retrieve_top_k(query, index, docs, model, k=5):
    q_embed = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_embed, k)
    return [docs[i] for i in I[0]]
