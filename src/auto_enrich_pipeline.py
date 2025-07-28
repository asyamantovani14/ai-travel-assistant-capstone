import json
import os
import spacy
from keybert import KeyBERT
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# === CONFIG ===
INPUT_FILE  = "data/blogs/clean_articles.json"
OUTPUT_FILE = "data/knowledge_base/family_travel_knowledge_enriched.json"

# === LOAD & CHECK ===
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

# === MODEL INITIALIZATION ===
# spaCy for NER
nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "lemmatizer"])

# KeyBERT for keywords
kw_model = KeyBERT("all-mpnet-base-v2")

# FLAN-T5 for advice extraction
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model  = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# === EXTRACTION FUNCTIONS ===
def extract_entities(text):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def extract_keyphrases(text, top_n=5):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )
    return [kw for kw, _ in keywords]

def extract_advice(text):
    prompt = (
        "Extract travel advice sentences from the text below:\n\n"
        f"{text}\n\n"
        "Return each tip on a new line."
    )
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = t5_model.generate(**inputs, max_new_tokens=150)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [line.strip() for line in decoded.split("\n") if line.strip()]

# === ENRICHMENT ===
enriched = []
for article in articles:
    text = article.get("clean_text", "")
    enriched.append({
        **article,
        "entities": extract_entities(text),
        "tags": extract_keyphrases(text),
        "tips": extract_advice(text)
    })

# === SAVE OUTPUT ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(enriched, f, indent=2, ensure_ascii=False)

print(f"✅ Enriched knowledge base saved to {OUTPUT_FILE}")
