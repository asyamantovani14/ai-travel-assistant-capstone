import spacy
import re
from word2number import w2n

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

    # Pattern esplicito: from X to Y
    pattern = re.search(r'from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+)', text, re.IGNORECASE)
    if pattern:
        origin = pattern.group(1).strip()
        destination = pattern.group(2).strip()
        destination = re.split(r'\b(for|with|and|on|in|at)\b', destination, 1)[0].strip()

        result["origin"] = origin
        result["destination"] = destination

    # Named entity recognition
    for ent in doc.ents:
        if ent.label_ == "GPE":
            if result["destination"] is None:
                result["destination"] = ent.text
            elif result["origin"] is None:
                result["origin"] = ent.text

        elif ent.label_ == "MONEY" and result["budget"] is None:
            try:
                result["budget"] = int(re.sub(r"[^\d]", "", ent.text))
            except:
                try:
                    result["budget"] = w2n.word_to_num(ent.text)
                except:
                    pass

        elif ent.label_ == "DATE" and result["duration"] is None:
            try:
                result["duration"] = int(re.search(r"\d+", ent.text).group())
            except:
                try:
                    result["duration"] = w2n.word_to_num(ent.text)
                except:
                    pass

        elif ent.label_ in ["NORP", "ORG"] and result["cuisine"] is None:
            if "food" in ent.text.lower() or "cuisine" in ent.text.lower():
                result["cuisine"] = ent.text

    # Heuristic check for cuisines
    if result["cuisine"] is None:
        for token in doc:
            if token.lemma_.lower() in ["italian", "mexican", "french", "thai", "japanese", "spanish", "indian", "greek"]:
                result["cuisine"] = token.lemma_.capitalize()
                break

    # Fallback: spelled-out budget detection outside MONEY tag
    if result["budget"] is None:
        try:
            budget_match = re.search(r"(?:spend|budget(?: of)?|around)\s+(.*?)(?:\s|dollars|usd)", text.lower())
            if budget_match:
                result["budget"] = w2n.word_to_num(budget_match.group(1))
        except:
            pass

    return result
