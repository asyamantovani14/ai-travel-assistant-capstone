import os
import sys
import pathlib
import re
import json
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import folium
import plotly.express as px
from collections import Counter

# ─── Setup path and env ────────────────────────────────────────────────
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ─── App Imports ────────────────────────────────────────────────────────
from rag_pipeline.generate_response import generate_response, generate_response_without_rag
from nlp.ner_utils import extract_entities
from utils.logger import log_interaction
from utils.csv_logger import save_response_to_csv
from utils.feedback_logger import save_feedback

# ─── Load index, docs, model ────────────────────────────────────────────
@st.cache_resource
def load_resources():
    index = faiss.read_index("data/indexes/travel_index.faiss")
    with open("data/indexes/docs_list.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, documents, model

index, all_documents, model = load_resources()

# ─── UI Layout ──────────────────────────────────────────────────────────
st.title(":globe_with_meridians: AI Travel Assistant")

query = st.text_input(":airplane: Enter your travel query")

st.subheader(":jigsaw: Optional Filters")
apply_country = st.checkbox("Filter by country")
allowed_countries = st.text_input("Countries (comma-separated)").lower().split(",") if apply_country else []

apply_duration = st.checkbox("Filter by trip duration")
min_days = st.number_input("Min days", min_value=1, step=1) if apply_duration else None
max_days = st.number_input("Max days", min_value=1, step=1) if apply_duration else None

apply_activities = st.checkbox("Filter by preferred activities")
activity_keywords = st.text_input("Activities (comma-separated)").lower().split(",") if apply_activities else []

apply_budget = st.checkbox("Set a max budget")
budget_limit = st.number_input("Budget in USD", min_value=0) if apply_budget else None

model_choice = st.selectbox(":robot_face: Select OpenAI model", ["gpt-3.5-turbo", "gpt-4"], index=0)

# ─── Map Function ───────────────────────────────────────────────────────
def show_itinerary_map(origin, destination):
    st.write(f"\n🗺️ Generating map from **{origin}** to **{destination}**...")
    try:
        geolocator = Nominatim(user_agent="travel-assistant-map")
        origin_loc = geolocator.geocode(origin, timeout=10)
        dest_loc = geolocator.geocode(destination, timeout=10)

        if not origin_loc:
            st.error(f"❌ Origin location not found: {origin}")
            return
        if not dest_loc:
            st.error(f"❌ Destination location not found: {destination}")
            return

        st.success(f"✅ Located **{origin}** at ({origin_loc.latitude:.2f}, {origin_loc.longitude:.2f})")
        st.success(f"✅ Located **{destination}** at ({dest_loc.latitude:.2f}, {dest_loc.longitude:.2f})")

        midpoint = [
            (origin_loc.latitude + dest_loc.latitude) / 2,
            (origin_loc.longitude + dest_loc.longitude) / 2
        ]

        m = folium.Map(location=midpoint, zoom_start=4)
        folium.Marker([origin_loc.latitude, origin_loc.longitude], popup=f"{origin} (Start)", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([dest_loc.latitude, dest_loc.longitude], popup=f"{destination} (End)", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine([(origin_loc.latitude, origin_loc.longitude), (dest_loc.latitude, dest_loc.longitude)], color="blue", weight=3).add_to(m)

        st_folium(m, width=700, height=500)
    except Exception as e:
        st.error(f"🌐 Map generation error: {e}")

# ─── Processing on submission ───
if "top_matches" not in st.session_state:
    st.session_state.top_matches = []
if "similarities" not in st.session_state:
    st.session_state.similarities = []
if "extracted_entities" not in st.session_state:
    st.session_state.extracted_entities = {}
if "rag_response" not in st.session_state:
    st.session_state.rag_response = ""
if "gpt_response" not in st.session_state:
    st.session_state.gpt_response = ""

if st.button(":mag: Get Itinerary") and query:
    # FILTRI + SIMILARITY
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
        st.info("Ranking documents by similarity...")

        query_embedding = model.encode([query]).astype("float32")
        embeddings = model.encode(filtered_docs).astype("float32")
        temp_index = faiss.IndexFlatL2(embeddings.shape[1])
        temp_index.add(embeddings)
        k = min(5, len(filtered_docs))
        distances, indices = temp_index.search(query_embedding, k)

        top_matches = [filtered_docs[idx] for idx in indices[0]]
        similarities = [1 / (1 + distances[0][i]) for i in range(k)]

        st.session_state.top_matches = top_matches
        st.session_state.similarities = similarities

        st.metric(label=":bar_chart: Avg Similarity", value=f"{np.mean(similarities):.4f}")
        st.metric(label=":arrow_down_small: Min Similarity", value=f"{np.min(similarities):.4f}")
        st.metric(label=":arrow_up_small: Max Similarity", value=f"{np.max(similarities):.4f}")

        with st.spinner("Generating RAG and GPT responses..."):
            st.session_state.rag_response = generate_response(query, top_matches, model=model_choice)
            st.session_state.gpt_response = generate_response_without_rag(query, model=model_choice)
            st.session_state.extracted_entities = extract_entities(query)

        log_interaction(
            query=query,
            matched_docs=top_matches,
            response=st.session_state.rag_response,
            extracted_entities=st.session_state.extracted_entities,
            model=model_choice,
            similarities=similarities
        )

        save_response_to_csv(
            query=query,
            response=st.session_state.rag_response,
            model=model_choice,
            entities=st.session_state.extracted_entities,
            prompt=None
        )

# ─── Mostra Risultati, Mappa, Feedback ───
if st.session_state.rag_response:
    st.subheader(":globe_with_meridians: Side-by-Side Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RAG-based Answer:**")
        st.markdown(st.session_state.rag_response)
    with col2:
        st.markdown("**GPT-only Answer:**")
        st.markdown(st.session_state.gpt_response)

    if st.session_state.extracted_entities.get("origin") and st.session_state.extracted_entities.get("destination"):
        with st.expander("Click to preview extracted locations"):
            st.json({
                "origin": st.session_state.extracted_entities["origin"],
                "destination": st.session_state.extracted_entities["destination"]
            })

        if st.button("🗺️ Show Map Itinerary"):
            show_itinerary_map(
                st.session_state.extracted_entities["origin"],
                st.session_state.extracted_entities["destination"]
            )

    st.subheader(":ballot_box_with_ballot: Feedback")
    choice = st.radio("Which response is better?", ["RAG-based", "GPT-only", "Both", "None"])
    notes = st.text_area("Explain your choice (optional)")
    if st.button("Submit Feedback"):
        save_feedback(query, st.session_state.rag_response, st.session_state.gpt_response, choice, notes)
        st.success("Feedback submitted.")
