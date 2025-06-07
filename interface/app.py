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

# Setup path and env
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# App Imports
from rag_pipeline.generate_response import generate_response, generate_response_without_rag
from nlp.ner_utils import extract_entities
from utils.logger import log_interaction
from utils.csv_logger import save_response_to_csv

import plotly.express as px
from collections import Counter

# Load resources
@st.cache_resource
def load_resources():
    index = faiss.read_index("data/indexes/travel_index.faiss")
    with open("data/indexes/docs_list.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, documents, model

index, all_documents, model = load_resources()

st.title("🌍 AI Travel Assistant")

query = st.text_input("✈️ Enter your travel query")

st.subheader("🯩 Optional Filters")
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

            top_embeddings = model.encode(top_matches).astype("float32")
            similarities = [1 / (1 + np.linalg.norm(query_embedding - emb.reshape(1, -1))) for emb in top_embeddings]

            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)

            st.metric(label="📊 Avg Similarity", value=f"{avg_sim:.4f}")
            st.metric(label="🔽 Min Similarity", value=f"{min_sim:.4f}")
            st.metric(label="🔼 Max Similarity", value=f"{max_sim:.4f}")

            extracted_entities = extract_entities(query)

            log_interaction(query, top_matches, answer, extracted_entities, model_choice, similarities)
            save_response_to_csv(query, answer, model_choice, extracted_entities, None)

        st.subheader("📎 GPT-only Comparison (no RAG)")
        with st.spinner("Generating GPT-only response..."):
            baseline = generate_response_without_rag(query, model=model_choice)
            st.info(baseline)

        st.subheader("🧠 Extracted Entities")
        st.json(extracted_entities)

        st.subheader("🗳️ Give Your Feedback")
        selected = st.radio("Which response is more helpful?", ["RAG-based", "LLM-only", "Both equally good", "None"], index=0)
        notes = st.text_area("Optional notes or justification", placeholder="Write your thoughts here...")

        if st.button("Submit Feedback"):
            from utils.feedback_logger import save_feedback
            save_feedback(query, answer, baseline, selected, notes)
            st.success("✅ Feedback saved successfully.")

st.markdown("## 💾 Saved Responses")
if os.path.exists("logs/responses.csv"):
    df = pd.read_csv("logs/responses.csv")
    df = df.dropna(subset=["query"])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date

    st.dataframe(df, use_container_width=True)

    if "destination" in df.columns:
        dest_freq = Counter(df["destination"].dropna())
        top_dest = pd.DataFrame(dest_freq.items(), columns=["Destination", "Count"]).sort_values("Count", ascending=False)
        st.subheader("Top Destinations")
        st.plotly_chart(px.bar(top_dest.head(10), x="Destination", y="Count"), use_container_width=True)

    if "date" in df.columns:
        df_time = df["date"].value_counts().sort_index()
        st.subheader("Query Frequency Over Time")
        st.plotly_chart(px.line(x=df_time.index, y=df_time.values, labels={"x": "Date", "y": "Number of Queries"}), use_container_width=True)

    if os.path.exists("logs/feedback.csv"):
        fdb = pd.read_csv("logs/feedback.csv")
        if "timestamp" in fdb.columns:
            fdb["timestamp"] = pd.to_datetime(fdb["timestamp"], errors="coerce")
            fdb["date"] = fdb["timestamp"].dt.date
            f_counts = fdb["date"].value_counts().sort_index()
            st.subheader("Feedback Over Time")
            st.plotly_chart(px.line(x=f_counts.index, y=f_counts.values, labels={"x": "Date", "y": "Feedback Count"}), use_container_width=True)

        if "preferred" in fdb.columns:
            pie = fdb["preferred"].value_counts().reset_index()
            pie.columns = ["Preference", "Count"]
            st.plotly_chart(px.pie(pie, names="Preference", values="Count"), use_container_width=True)

    st.download_button("Download CSV", df.to_csv(index=False).encode(), "responses.csv")
    if os.path.exists("logs/query_log.txt"):
        with open("logs/query_log.txt", "r", encoding="utf-8") as f:
            st.download_button("Download Full Log TXT", f.read(), "query_log.txt")
else:
    st.info("No responses saved yet.")
