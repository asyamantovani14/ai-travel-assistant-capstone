"""Application service for retrieval-backed conversational travel planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from nlp.ner_utils import extract_entities
from rag_pipeline.generate_response import generate_response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "travel_index.faiss"
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "indexes" / "docs_list.json"
URL_RE = re.compile(r"https?://[^\s)<>\"]+")


@dataclass(frozen=True)
class TravelFilters:
    countries: tuple[str, ...] = ()
    activities: tuple[str, ...] = ()
    min_days: int | None = None
    max_budget: int | None = None

    @property
    def active(self) -> bool:
        return bool(self.countries or self.activities or self.min_days or self.max_budget)


def document_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        return "\n".join(
            str(document.get(key, "")) for key in ("title", "text", "url")
        )
    return str(document)


def matches_filters(document: Any, filters: TravelFilters) -> bool:
    text = document_text(document).lower()
    if filters.countries and not any(value in text for value in filters.countries):
        return False
    if filters.activities and not any(value in text for value in filters.activities):
        return False
    if filters.min_days:
        duration = re.search(r"(\d+)\s*[- ]?days?", text)
        if duration and int(duration.group(1)) < filters.min_days:
            return False
    if filters.max_budget:
        prices = [int(value) for value in re.findall(r"\$\s?(\d{2,6})", text)]
        if prices and min(prices) > filters.max_budget:
            return False
    return True


class TravelService:
    def __init__(self, index, documents, embedding_model):
        if index.ntotal != len(documents):
            raise ValueError("Travel index and document list are out of sync")
        self.index = index
        self.documents = documents
        self.embedding_model = embedding_model

    @classmethod
    def load(cls):
        if not INDEX_PATH.exists() or not DOCUMENTS_PATH.exists():
            raise FileNotFoundError("Travel index is missing; run the index build pipeline")
        index = faiss.read_index(str(INDEX_PATH))
        with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
            documents = json.load(file)
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        return cls(index, documents, model)

    def retrieve(self, query: str, filters: TravelFilters, k: int = 5):
        query_vector = self.embedding_model.encode([query]).astype("float32")
        if not filters.active:
            distances, indices = self.index.search(
                query_vector, min(k, self.index.ntotal)
            )
            return [
                (self.documents[position], float(1 / (1 + distance)))
                for position, distance in zip(indices[0], distances[0])
                if position >= 0
            ]

        candidates = [
            document
            for document in self.documents
            if matches_filters(document, filters)
        ]
        if not candidates:
            return []
        texts = [document_text(document) for document in candidates]
        embeddings = self.embedding_model.encode(texts).astype("float32")
        filtered_index = faiss.IndexFlatL2(embeddings.shape[1])
        filtered_index.add(embeddings)
        distances, indices = filtered_index.search(
            query_vector, min(k, len(candidates))
        )
        return [
            (candidates[position], float(1 / (1 + distance)))
            for position, distance in zip(indices[0], distances[0])
            if position >= 0
        ]

    def plan(self, message, history, filters):
        recent_user_turns = [
            item.get("content", "")
            for item in history
            if item.get("role") == "user"
        ][-2:]
        retrieval_query = " ".join(recent_user_turns + [message])
        matches = self.retrieve(retrieval_query, filters)
        if not matches:
            return {
                "answer": "I could not find travel knowledge matching those filters. Try broadening them.",
                "sources": [],
                "trip": extract_entities(retrieval_query),
            }

        documents = [document_text(document) for document, _ in matches]
        answer = generate_response(
            message,
            documents,
            conversation_history=history,
        )
        sources = []
        for document, score in matches:
            text = document_text(document)
            url_match = URL_RE.search(text)
            title = text.strip().splitlines()[0][:120] or "Travel source"
            sources.append(
                {
                    "title": title,
                    "excerpt": text[:480],
                    "url": url_match.group(0) if url_match else None,
                    "score": round(score, 4),
                }
            )
        trip = {
            key: value
            for key, value in extract_entities(retrieval_query).items()
            if value not in (None, "", "NA")
        }
        return {"answer": answer, "sources": sources, "trip": trip}


@lru_cache(maxsize=1)
def get_travel_service():
    return TravelService.load()
