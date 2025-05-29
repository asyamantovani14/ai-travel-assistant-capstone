import spacy
import pickle
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load your documents
with open("data/index/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Extract named entities from each document
extracted_entities = []

for doc_text in tqdm(documents, desc="Extracting entities"):
    doc = nlp(doc_text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    extracted_entities.append({
        "text": doc_text,
        "entities": ents
    })

# Save to disk
with open("data/index/entities.pkl", "wb") as f:
    pickle.dump(extracted_entities, f)

print("Entity extraction completed and saved.")
