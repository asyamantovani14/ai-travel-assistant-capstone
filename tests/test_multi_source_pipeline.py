from data_loader.multi_loader import load_all_sources
from rag_pipeline.document_retriever import build_faiss_index, retrieve_top_k
from rag_pipeline.generate_response import generate_response

# 1. Carica tutti i documenti
all_docs = load_all_sources()

# 2. Costruisci l'indice FAISS
index, _, model = build_faiss_index(all_docs)

# 3. Fai una query di test
query = "Plan a romantic road trip in Italy with pet-friendly stays"
top_docs = retrieve_top_k(query, index, all_docs, model)

# 4. Genera la risposta
response = generate_response(query, top_docs)
print(response)


