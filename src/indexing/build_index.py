import os
import sys
import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Aggiunge src/ al path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_loader.multi_loader import load_all_sources

def build_index():
    print("🔄 Caricamento documenti da più fonti...")
    docs = load_all_sources("data/sources")

    print(f"📚 Totale documenti caricati: {len(docs)}")

    print("⚙️ Generazione degli embedding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True).astype("float32")

    print("🧠 Costruzione dell'indice FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("💾 Salvataggio dell'indice e dei documenti...")
    os.makedirs("data/indexes", exist_ok=True)
    faiss.write_index(index, "data/indexes/travel_index.faiss")
    with open("data/indexes/docs_list.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print("✅ Indice FAISS creato con successo!")

if __name__ == "__main__":
    build_index()
