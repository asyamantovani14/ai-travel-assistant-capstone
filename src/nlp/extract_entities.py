import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extracts structured entities from a travel query.

    Returns a dictionary with keys like 'origin', 'destination', 'budget', etc.
    """
    doc = nlp(text)
    
    result = {
        "origin": None,
        "destination": None,
        "cuisine": None,
        "budget": None,
        "duration": None
    }

    for ent in doc.ents:
        if ent.label_ == "GPE":
            if result["destination"] is None:
                result["destination"] = ent.text
            elif result["origin"] is None:
                result["origin"] = ent.text

        elif ent.label_ == "MONEY":
            match = re.search(r"\d+", ent.text.replace(",", ""))
            if match:
                result["budget"] = int(match.group(0))

        elif ent.label_ == "DATE":
            match = re.search(r"\d+", ent.text)
            if match:
                result["duration"] = int(match.group(0))

        elif ent.label_ == "NORP" or ent.label_ == "ORG":  # fallback per cucine
            if "food" in ent.text.lower() or "cuisine" in ent.text.lower():
                result["cuisine"] = ent.text

    # Keywords manuali per cucina (es. "Spanish food")
    if result["cuisine"] is None:
        for token in doc:
            if token.lemma_.lower() in ["italian", "mexican", "french", "thai", "japanese", "spanish"]:
                result["cuisine"] = token.lemma_.capitalize()
                break

    return result
