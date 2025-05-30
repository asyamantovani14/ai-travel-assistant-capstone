import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_entities(query: str) -> dict:
    doc = nlp(query)

    origin = None
    destination = None
    budget = None
    cuisine = None
    duration = None
    activities = []

    # Named entity recognition
    for ent in doc.ents:
        if ent.label_ == "GPE":
            # Per ora il primo GPE lo consideriamo come destinazione
            if not destination:
                destination = ent.text
            elif not origin:
                origin = ent.text
        elif ent.label_ == "MONEY":
            match = re.search(r"\d+", ent.text.replace(",", ""))
            if match:
                budget = int(match.group())
        elif ent.label_ == "DATE":
            match = re.search(r"\d+", ent.text)
            if match:
                duration = int(match.group())

    # Rule-based parsing per cucina o attività
    cuisine_keywords = ["italian", "mexican", "japanese", "chinese", "spanish", "thai", "indian"]
    activity_keywords = ["beach", "hiking", "ski", "museum", "culture", "adventure", "relax", "shopping", "food"]

    for token in doc:
        word = token.text.lower()
        if word in cuisine_keywords:
            cuisine = word
        if word in activity_keywords:
            activities.append(word)

    return {
        "origin": origin,
        "destination": destination,
        "cuisine": cuisine,
        "budget": budget,
        "duration": duration,
        "activities": list(set(activities))  # rimuove duplicati
    }
