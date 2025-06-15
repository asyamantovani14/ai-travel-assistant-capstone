import os
import sys
import pathlib
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Ensure src/ is in sys.path for absolute imports
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import log_interaction
from nlp.ner_utils import extract_entities
from agents.tool_wrappers import generate_smart_enrichment
from utils.csv_logger import save_response_to_csv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def format_response_markdown(text):
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    return "\n\n---\n\n".join(blocks)


def generate_response(query, context_docs, model="gpt-3.5-turbo", client=None, log_dir="logs"):
    try:
        if client is None:
            client = OpenAI(api_key=api_key)

        extracted_entities = extract_entities(query)
        tool_context = generate_smart_enrichment(extracted_entities)
        truncated_docs = [doc[:1000] for doc in context_docs]
        context = tool_context + "\n\n" + "\n\n".join(truncated_docs)

        # Prompt migliorato
        final_prompt = f"""
You are a professional and friendly travel assistant specialized in customized travel plans.

Your task is to create a helpful, concise and creative travel suggestion for the user's query using the contextual documents and additional tools.

Please include:
- An engaging itinerary based on location and duration
- Relevant restaurant or hotel tips if available
- Specific elements matching the user's preferences (e.g. food, budget, activities)

Context:
{context}

User Query:
{query}

Respond in a markdown-friendly format.
Answer:
"""

        # Sistema di tono dinamico
        if "adventure" in query.lower() or "hiking" in query.lower():
            tone = "You are an energetic and adventurous travel assistant who gives thrilling suggestions."
        else:
            tone = "You are a calm and professional travel assistant who provides relaxing and well-organized suggestions."

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": tone},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        raw_response = response.choices[0].message.content.strip()
        final_response = format_response_markdown(raw_response)

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
