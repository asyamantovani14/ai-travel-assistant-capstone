import os
import sys
import pathlib
import json
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Ensure src/ is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import log_interaction
from nlp.ner_utils import extract_entities
from agents.tool_wrappers import generate_smart_enrichment
from utils.csv_logger import save_response_to_csv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Path to enriched knowledge base
KB_FILE = os.path.join(ROOT_DIR, "data", "knowledge_base", "family_travel_knowledge_enriched.json")


def format_response_markdown(text):
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    return "\n\n---\n\n".join(blocks)


def load_enriched_kb():
    """Load enriched knowledge base JSON"""
    if not os.path.exists(KB_FILE):
        return []
    with open(KB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_kb_snippet(kb_entries):
    """Build a brief snippet from enriched KB entries"""
    snippet = []
    for e in kb_entries:
        title = e.get("title")
        tips = e.get("tips", [])[:2]
        tags = e.get("tags", [])[:5]
        entry_lines = [f"**{title}**"]
        for t in tips:
            entry_lines.append(f"- {t}")
        if tags:
            entry_lines.append(f"Tags: {', '.join(tags)}")
        snippet.append("\n".join(entry_lines))
        if len(snippet) >= 3:
            break
    return "\n\n".join(snippet)


def generate_response(query, context_docs, model="gpt-3.5-turbo", client=None, log_dir="logs"):
    if client is None:
        client = OpenAI(api_key=api_key)

    # Extract entities and tool context
    extracted_entities = extract_entities(query)
    tool_context = generate_smart_enrichment(extracted_entities)

    # Load enriched KB and build snippet
    kb_entries = load_enriched_kb()
    kb_snippet = build_kb_snippet(kb_entries)

    # Prepare context docs
    truncated_docs = [doc[:1000] for doc in context_docs if doc.strip()]
    has_human_opinion = bool(kb_entries)

    # Assemble prompt
    prompt_parts = [
        "You are a professional and friendly travel assistant.",
        "", 
        "Your task is to generate a customized, engaging travel itinerary using the user query and blog-based documents.",
        "", 
        "Instructions:",
        "- Create a daily itinerary if possible.",
        "- Use **direct quotes** from the blogs with **citations** (e.g. [source](https://blog.com/post)).",
        "- Include practical tips and tags extracted from the blogs.",
        "- If no relevant blog data exists, explain clearly there are no blog-based human opinions.",
        "- Keep the format clean and Markdown-friendly.",
        "- Add a final note suggesting the user ask again for other destinations or options.",
        ""
    ]
    if kb_snippet:
        prompt_parts.append("Blog Enriched Data:")
        prompt_parts.append(kb_snippet)
        prompt_parts.append("")
    prompt_parts.append("Context from tools and blogs:")
    prompt_parts.append(tool_context)
    prompt_parts.extend(truncated_docs)
    prompt_parts.append("")
    prompt_parts.append(f"User Query:\n{query}")
    prompt_parts.append("")
    prompt_parts.append("Answer (markdown format):")
    final_prompt = "\n".join(prompt_parts)

    # Tone
    if any(k in query.lower() for k in ["family", "adventure", "backpack"]):
        tone = "You are an energetic and curious travel expert specializing in family trips, adventures and relaxing getaways."
    else:
        tone = "You are a calm and thoughtful travel planner with a focus on high-quality suggestions."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": tone},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.8,
            max_tokens=700
        )

        raw_response = response.choices[0].message.content.strip()
        final_response = format_response_markdown(raw_response)

        if not has_human_opinion:
            final_response += (
                "\n\n---\n\n"
                "⚠️ We couldn't find any relevant blog-based human opinions for your query. "
                "Try refining your question or exploring a different destination."
            )

        log_interaction(
            query=query,
            matched_docs=context_docs,
            response=final_response,
            extracted_entities=extracted_entities,
            model=model,
            final_prompt=final_prompt,
            log_dir=log_dir
        )

        save_response_to_csv(
            query=query,
            response=final_response,
            model=model,
            entities=extracted_entities,
            prompt=final_prompt
        )

        st.download_button("📋 Copy Response", data=final_response, file_name="response.txt")
        st.code(final_response, language="markdown")

        return final_response

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        log_interaction(
            query=query,
            matched_docs=context_docs,
            response=error_msg,
            model=model,
            log_dir=log_dir
        )
        return error_msg


def generate_response_without_rag(query, model="gpt-3.5-turbo", client=None):
    if client is None:
        client = OpenAI(api_key=api_key)
    try:
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": query}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        raw_response = response.choices[0].message.content.strip()
        final_response = format_response_markdown(raw_response)

        st.download_button("📋 Copy GPT-only Response", data=final_response, file_name="gpt_response.txt")
        st.code(final_response, language="markdown")

        return final_response
    except Exception as e:
        return f"Error (no RAG): {e}"
