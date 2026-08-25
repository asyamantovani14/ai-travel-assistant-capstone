import json
import os
import pathlib
import re
import sys

import faiss
import folium
import numpy as np
import streamlit as st
from geopy.geocoders import Nominatim
from sentence_transformers import SentenceTransformer
from streamlit_folium import st_folium


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from nlp.ner_utils import extract_entities
from rag_pipeline.generate_response import generate_response


st.set_page_config(page_title="Travel Assistant", page_icon="🌍", layout="wide")


@st.cache_resource(show_spinner="Loading travel knowledge...")
def load_resources():
    index_path = ROOT_DIR / "data" / "indexes" / "travel_index.faiss"
    docs_path = ROOT_DIR / "data" / "indexes" / "docs_list.json"
    if not index_path.exists() or not docs_path.exists():
        raise FileNotFoundError(
            "Travel index not found. Run: python src/indexing/build_index.py"
        )
    index = faiss.read_index(str(index_path))
    with docs_path.open("r", encoding="utf-8") as file:
        documents = json.load(file)
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    if index.ntotal != len(documents):
        raise ValueError("The travel index and document list are out of sync.")
    return index, documents, model


def document_text(document):
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        return "\n".join(
            str(document.get(key, "")) for key in ("title", "text", "url")
        )
    return str(document)


def matches_filters(document, countries, activities, min_days, max_budget):
    text = document_text(document).lower()
    if countries and not any(country in text for country in countries):
        return False
    if activities and not any(activity in text for activity in activities):
        return False
    if min_days:
        duration = re.search(r"(\d+)\s*[- ]?days?", text)
        if duration and int(duration.group(1)) < min_days:
            return False
    if max_budget:
        prices = [int(value) for value in re.findall(r"\$\s?(\d{2,6})", text)]
        if prices and min(prices) > max_budget:
            return False
    return True


def retrieve_documents(query, index, documents, model, filters, k=5):
    query_vector = model.encode([query]).astype("float32")
    has_filters = any(filters.values())

    if not has_filters:
        distances, indices = index.search(query_vector, min(k, index.ntotal))
        pairs = [
            (documents[position], float(1 / (1 + distance)))
            for position, distance in zip(indices[0], distances[0])
            if position >= 0
        ]
        return pairs

    candidates = [
        document
        for document in documents
        if matches_filters(document, **filters)
    ]
    if not candidates:
        return []
    texts = [document_text(document) for document in candidates]
    embeddings = model.encode(texts).astype("float32")
    filtered_index = faiss.IndexFlatL2(embeddings.shape[1])
    filtered_index.add(embeddings)
    distances, indices = filtered_index.search(query_vector, min(k, len(candidates)))
    return [
        (candidates[position], float(1 / (1 + distance)))
        for position, distance in zip(indices[0], distances[0])
        if position >= 0
    ]


def conversation_markdown(messages):
    sections = []
    for message in messages:
        heading = "You" if message["role"] == "user" else "Travel Assistant"
        sections.append(f"## {heading}\n\n{message['content']}")
    return "\n\n".join(sections)


def render_map(origin, destination):
    geolocator = Nominatim(user_agent="travel-assistant-map")
    origin_location = geolocator.geocode(origin, timeout=10)
    destination_location = geolocator.geocode(destination, timeout=10)
    if not origin_location or not destination_location:
        st.warning("One of the locations could not be found on the map.")
        return
    points = [
        (origin_location.latitude, origin_location.longitude),
        (destination_location.latitude, destination_location.longitude),
    ]
    midpoint = [sum(point[0] for point in points) / 2, sum(point[1] for point in points) / 2]
    itinerary_map = folium.Map(location=midpoint, zoom_start=5)
    folium.Marker(points[0], tooltip=origin).add_to(itinerary_map)
    folium.Marker(points[1], tooltip=destination).add_to(itinerary_map)
    folium.PolyLine(points, color="#146c5a", weight=4).add_to(itinerary_map)
    st_folium(itinerary_map, width=None, height=420, use_container_width=True)


def parse_csv_filter(value):
    return [part.strip().lower() for part in value.split(",") if part.strip()]


try:
    travel_index, all_documents, embedding_model = load_resources()
except Exception as error:
    st.error(str(error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Where would you like to go? Tell me your dates, budget, pace, and who is travelling.",
        }
    ]
if "trip_entities" not in st.session_state:
    st.session_state.trip_entities = {}

with st.sidebar:
    st.header("Trip preferences")
    countries_value = st.text_input("Countries", placeholder="Italy, France")
    activities_value = st.text_input("Activities", placeholder="Hiking, museums")
    min_days_value = st.number_input("Minimum days", min_value=0, value=0, step=1)
    max_budget_value = st.number_input("Maximum budget (USD)", min_value=0, value=0, step=100)
    st.divider()
    if st.button("New trip", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.trip_entities = {}
        st.rerun()
    st.download_button(
        "Download conversation",
        data=conversation_markdown(st.session_state.messages),
        file_name="travel-plan.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.caption(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')}")

st.title("Travel Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Travel knowledge used"):
                for position, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**{position}. Relevance {source['score']:.3f}**")
                    st.caption(source["text"][:700])

entities = st.session_state.trip_entities
origin = entities.get("origin")
destination = entities.get("destination")
if origin and destination:
    with st.expander(f"Map: {origin} to {destination}"):
        render_map(origin, destination)

query = st.chat_input("Plan a trip or refine your itinerary")
if query:
    prior_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    recent_user_turns = [
        message["content"] for message in prior_messages if message["role"] == "user"
    ][-2:]
    retrieval_query = " ".join(recent_user_turns + [query])
    filters = {
        "countries": parse_csv_filter(countries_value),
        "activities": parse_csv_filter(activities_value),
        "min_days": min_days_value or None,
        "max_budget": max_budget_value or None,
    }

    with st.chat_message("assistant"):
        with st.spinner("Building your itinerary..."):
            matches = retrieve_documents(
                retrieval_query,
                travel_index,
                all_documents,
                embedding_model,
                filters,
            )
            if not matches:
                answer = "I could not find travel knowledge matching those filters. Try broadening them."
            else:
                documents = [document_text(document) for document, _ in matches]
                answer = generate_response(
                    query,
                    documents,
                    conversation_history=prior_messages,
                )
            st.markdown(answer)

    sources = [
        {"text": document_text(document), "score": score}
        for document, score in matches
    ]
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    extracted = extract_entities(retrieval_query)
    st.session_state.trip_entities = {
        key: value
        for key, value in extracted.items()
        if value not in (None, "", "NA")
    }
    st.rerun()
