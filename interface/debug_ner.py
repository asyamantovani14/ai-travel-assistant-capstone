import sys
import os
import pathlib

# Aggiunge src/ al PYTHONPATH
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
from nlp.ner_utils import extract_entities


st.title("🔍 Entity Extraction Debugger")

query = st.text_input("Enter a travel query to extract entities:")

if query:
    entities = extract_entities(query)
    st.subheader("📌 Extracted Entities")
    st.json(entities)

    st.subheader("🧠 Raw SpaCy Entities")
    doc = extract_entities.__globals__["nlp"](query)
    st.write([(ent.text, ent.label_) for ent in doc.ents])
