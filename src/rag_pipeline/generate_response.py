# generate_response.py

import os
import sys
import pathlib
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

def format_response_markdown(text):
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    return "\n\n---\n\n".join(blocks)

def generate_response(query, context_docs, model="gpt-3.5-turbo", client=None, log_dir="logs"):
    if client is None:
        client = OpenAI(api_key=api_key)

    extracted_entities = extract_entities(query)
    tool_context = generate_smart_enrichment(extracted_entities)
    truncated_docs = [doc[:1000] for doc in context_docs if doc.strip()]
    context = tool_context + "\n\n" + "\n\n".join(truncated_docs)

    has_human_opinion = any("http" in doc for doc in context_docs)

    final_prompt = f"""
You are a professional and friendly travel assistant.

Your task is to generate a customized, engaging travel itinerary using the user query and the blog-based documents.

Instructions:
- Create a daily itinerary if possible.
- Use **direct quotes** from the blogs if available.
- Add **citations with hyperlinks** (e.g. `[source](https://blog.com/post123)`).
- If no relevant blog data exists, explain clearly that there are no blog-based human opinions.
- Keep the format clean and Markdown-friendly.
- Add a final note to suggest the user ask again for other destinations or options.

Context from tools and blogs:
{context}

User Query:
{query}

Answer (markdown format):
"""

    tone = (
        "You are an energetic and curious travel expert specializing in family trips, adventures and relaxing getaways."
        if "family" in query.lower() or "adventure" in query.lower()
        else "You are a calm and thoughtful travel planner with a focus on high-quality suggestions."
    )

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
