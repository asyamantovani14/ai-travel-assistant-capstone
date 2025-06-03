import spacy
import re

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extracts structured information from a travel-related query.

    Args:
        text (str): User input query.

    Returns:
        dict: Dictionary with keys like origin, destination, cuisine, budget, duration.
    """
    doc = nlp(text)

    result = {
        "origin": None,
        "destination": None,
        "cuisine": None,
        "budget": None,
        "duration": None
    }

    # Named entity recognition
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

        elif ent.label_ in ["NORP", "ORG"]:
            if "food" in ent.text.lower() or "cuisine" in ent.text.lower():
                result["cuisine"] = ent.text

    # Heuristic check for cuisines
    if result["cuisine"] is None:
        for token in doc:
            if token.lemma_.lower() in ["italian", "mexican", "french", "thai", "japanese", "spanish", "indian", "greek"]:
                result["cuisine"] = token.lemma_.capitalize()
                break

    return result
