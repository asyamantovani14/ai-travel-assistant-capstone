import sys
import os
import pathlib

# Aggiunge src/ al PYTHONPATH
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import faiss
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from rag_pipeline.generate_response import generate_response, generate_response_without_rag
from nlp.ner_utils import extract_entities

# Load index and documents
@st.cache_resource
def load_resources():
    index = faiss.read_index("data/index/travel_index.faiss")
    with open("data/index/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, documents, model

index, all_documents, model = load_resources()

st.title("🌍 AI Travel Assistant")

# User input
query = st.text_input("✈️ Enter your travel query")

# Optional Filters
st.subheader("🧩 Filters (optional)")

apply_country = st.checkbox("Filter by country")
if apply_country:
    allowed_countries = st.text_input("Countries (comma separated)").lower().split(',')

apply_duration = st.checkbox("Filter by trip duration")
if apply_duration:
    min_days = st.number_input("Min days", min_value=1, step=1)
    max_days = st.number_input("Max days", min_value=1, step=1)

apply_activities = st.checkbox("Filter by preferred activities")
if apply_activities:
    activity_keywords = st.text_input("Activities (comma separated)").lower().split(',')

apply_budget = st.checkbox("Set a max budget")
if apply_budget:
    budget_limit = st.number_input("Budget in USD", min_value=0)

# Submit
if st.button("🔍 Get Itinerary") and query:
    filtered_docs = []
    for doc in all_documents:
        doc_lower = doc.lower()
        include = True

        if apply_country:
            include &= any(c.strip() in doc_lower for c in allowed_countries)

        if apply_duration:
            match = re.search(r"(\d+)\s*[- ]?day[s]?", doc_lower)
            if match:
                days = int(match.group(1))
                include &= min_days <= days <= max_days
            else:
                include = False

        if apply_activities:
            include &= any(a.strip() in doc_lower for a in activity_keywords)

        if apply_budget:
            match = re.search(r"\$?(\d{3,5})", doc)
            if match:
                include &= float(match.group(1)) <= budget_limit
            else:
                include = False

        if include:
            filtered_docs.append(doc)

    if not filtered_docs:
        st.warning("No documents match the filters. Try relaxing them.")
    else:
        st.info("Ranking documents...")
        query_embedding = model.encode([query]).astype("float32")
        embeddings = model.encode(filtered_docs).astype("float32")
        temp_index = faiss.IndexFlatL2(embeddings.shape[1])
        temp_index.add(embeddings)
        k = min(5, len(filtered_docs))
        distances, indices = temp_index.search(query_embedding, k)

        top_matches = []
        for i, idx in enumerate(indices[0]):
            sim_score = 1 / (1 + distances[0][i])
            top_matches.append(filtered_docs[idx])
            st.markdown(f"**Match {i+1}** (score: {sim_score:.4f})")
            st.write(filtered_docs[idx])
            st.divider()

        # Generate response
        st.subheader("🤖 Travel Assistant Suggestion")
        with st.spinner("Generating response..."):
            answer = generate_response(query, top_matches)
            st.success(answer)

            # Confronto con risposta senza RAG
            st.subheader("📎 Comparison with ChatGPT-only Response (no RAG)")
            with st.spinner("Generating standard ChatGPT reply..."):
                baseline_answer = generate_response_without_rag(query)
            st.info(baseline_answer)

        # Show extracted entities from query
        st.subheader("🔍 Extracted Entities (from query)")
        st.json(extract_entities(query))
