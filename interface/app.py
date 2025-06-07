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

# ─── Setup path and env ─────────────────────────────────────────────────────────
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ─── App Imports ────────────────────────────────────────────────────────────────
from rag_pipeline.generate_response import generate_response, generate_response_without_rag
from nlp.ner_utils import extract_entities
from utils.logger import log_interaction
from utils.csv_logger import save_response_to_csv

import plotly.express as px
from collections import Counter

# ─── Load index, docs, model ────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    index = faiss.read_index("data/indexes/travel_index.faiss")
    with open("data/indexes/docs_list.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, documents, model

index, all_documents, model = load_resources()

# ─── UI Layout ───────────────────────────────────────────────────────────────────
st.title("🌍 AI Travel Assistant")

query = st.text_input("✈️ Enter your travel query")

st.subheader("🧩 Optional Filters")
apply_country = st.checkbox("Filter by country")
allowed_countries = st.text_input("Countries (comma-separated)").lower().split(",") if apply_country else []

apply_duration = st.checkbox("Filter by trip duration")
min_days = st.number_input("Min days", min_value=1, step=1) if apply_duration else None
max_days = st.number_input("Max days", min_value=1, step=1) if apply_duration else None

apply_activities = st.checkbox("Filter by preferred activities")
activity_keywords = st.text_input("Activities (comma-separated)").lower().split(",") if apply_activities else []

apply_budget = st.checkbox("Set a max budget")
budget_limit = st.number_input("Budget in USD", min_value=0) if apply_budget else None

model_choice = st.selectbox("🤖 Select OpenAI model", ["gpt-3.5-turbo", "gpt-4"], index=0)

# ─── Processing on submission ───────────────────────────────────────────────────
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
        st.info("Ranking documents by similarity...")

        query_embedding = model.encode([query]).astype("float32")
        embeddings = model.encode(filtered_docs).astype("float32")
        temp_index = faiss.IndexFlatL2(embeddings.shape[1])
        temp_index.add(embeddings)

        k = min(5, len(filtered_docs))
        distances, indices = temp_index.search(query_embedding, k)

        top_matches = []
        for i, idx in enumerate(indices[0]):
            sim_score = 1 / (1 + distances[0][i])
            match_text = filtered_docs[idx]
            top_matches.append(match_text)
            st.markdown(f"**Match {i+1}** (score: {sim_score:.4f})")
            st.write(match_text)
            st.divider()

        st.subheader("🤖 Assistant Suggestion")
        with st.spinner("Generating answer with RAG..."):
            answer = generate_response(query, top_matches, model=model_choice)
            st.success(answer)

            query_embedding = model.encode([query]).astype("float32")
            top_embeddings = model.encode(top_matches).astype("float32")
            similarities = [1 / (1 + np.linalg.norm(query_embedding - emb.reshape(1, -1))) for emb in top_embeddings]

            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)

            st.metric(label="📊 Avg Similarity", value=f"{avg_sim:.4f}")
            st.metric(label="🔽 Min Similarity", value=f"{min_sim:.4f}")
            st.metric(label="🔼 Max Similarity", value=f"{max_sim:.4f}")

            extracted_entities = extract_entities(query)

            log_interaction(
                query=query,
                matched_docs=top_matches,
                response=answer,
                extracted_entities=extracted_entities,
                model=model_choice,
                similarities=similarities
            )
            save_response_to_csv(
                query=query,
                response=answer,
                model=model_choice,
                entities=extracted_entities,
                prompt=None
            )

        st.subheader("📎 GPT-only Comparison (no RAG)")
        with st.spinner("Generating GPT-only response..."):
            baseline = generate_response_without_rag(query, model=model_choice)
            st.info(baseline)

        st.subheader("🧠 Extracted Entities")
        st.json(extracted_entities)

# ─── Section: Saved Responses ───────────────────────────────────────────────────
st.markdown("## 📁 Saved Responses")
if os.path.exists("logs/responses.csv"):
    df = pd.read_csv("logs/responses.csv")

    search_text = st.text_input("🔎 Filter by query or model")
    if search_text:
        df = df[
            df["query"].str.contains(search_text, case=False, na=False)
            | df["model"].str.contains(search_text, case=False, na=False)
            | df["response"].str.contains(search_text, case=False, na=False)
        ]

    # Filtri entità (solo se esistono nel dataframe)
    with st.expander("🧪 Filter by extracted entities"):
        if "destination" in df.columns:
            selected_dest = st.selectbox("Destination", options=[""] + sorted(df["destination"].dropna().unique().tolist()))
            if selected_dest:
                df = df[df["destination"] == selected_dest]

        if "cuisine" in df.columns:
            selected_cuisine = st.selectbox("Cuisine", options=[""] + sorted(df["cuisine"].dropna().unique().tolist()))
            if selected_cuisine:
                df = df[df["cuisine"] == selected_cuisine]

        if "budget" in df.columns:
            max_budget = st.number_input("Maximum Budget", min_value=0)
            if max_budget:
                df = df[df["budget"].fillna(0) <= max_budget]



    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="saved_responses.csv", mime="text/csv")

    json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
    st.download_button("⬇️ Download JSON", data=json_bytes, file_name="saved_responses.json", mime="application/json")

    if os.path.exists("logs/query_log.txt"):
        with open("logs/query_log.txt", "r", encoding="utf-8") as f:
            log_txt = f.read()
        st.download_button("📄 Export Full Log (TXT)", data=log_txt, file_name="full_query_log.txt", mime="text/plain")

    if "destination" in df.columns:
        dest_counts = df["destination"].dropna()
        dest_freq = Counter(dest_counts)
        top_dest = pd.DataFrame(dest_freq.items(), columns=["Destination", "Count"]).sort_values(by="Count", ascending=False)

        st.subheader("📊 Top Destinations")
        fig = px.bar(top_dest.head(10), x="Destination", y="Count", title="Most Requested Destinations")
        st.plotly_chart(fig, use_container_width=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.date

        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()

        date_range = st.date_input("📅 Filter by date range", [min_date, max_date])
        if len(date_range) == 2:
            df = df[(df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])]


        st.subheader("📈 Query Frequency Over Time")
        time_counts = df["date"].value_counts().sort_index()
        fig_time = px.line(
            x=time_counts.index,
            y=time_counts.values,
            labels={"x": "Date", "y": "Number of Queries"},
            title="Query Frequency Over Time"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🧹 Maintenance")
    if st.button("Reset Saved Responses (clear logs/responses.csv)"):
        os.remove("logs/responses.csv")
        st.success("File logs/responses.csv eliminato.")
        st.experimental_rerun()
else:
    st.info("No responses saved yet.")
