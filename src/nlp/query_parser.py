import spacy
import re

nlp = spacy.load("en_core_web_sm")

def parse_query(query):
    doc = nlp(query)

    destination = duration = intent = budget = None
    activities = []

    for ent in doc.ents:
        if ent.label_ == "GPE":
            destination = ent.text
        elif ent.label_ == "MONEY":
            match = re.search(r"\d+", ent.text.replace(",", ""))
            if match:
                budget = int(match.group(0))
        elif ent.label_ == "DATE":
            match = re.search(r"\d+", ent.text)
            if match:
                duration = int(match.group(0))

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in ["relax", "adventure", "culture", "romantic", "nature"]:
            intent = lemma
        elif lemma in ["beach", "museum", "hike", "ski", "food"]:
            activities.append(lemma)

    return {
        "destination": destination,
        "duration": duration,
        "intent": intent,
        "activities": list(set(activities)),
        "budget": budget
    }
