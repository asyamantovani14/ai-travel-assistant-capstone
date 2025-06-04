import os
import sys
import pathlib
from dotenv import load_dotenv
from openai import OpenAI

# Ensure src/ is in sys.path for absolute imports
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import log_interaction
from nlp.ner_utils import extract_entities
from agents.tool_wrappers import (
    mock_google_maps_route,
    mock_restaurant_recommendation,
    mock_hotel_suggestions
)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def enrich_prompt_with_tools(query, entities):
    prompt = ""

    # Verifica che le entità siano già in formato dizionario {label: text}
    ent_dict = entities

    if ent_dict.get("origin") and ent_dict.get("destination"):
        prompt += mock_google_maps_route(ent_dict["origin"], ent_dict["destination"]) + "\n"

    if ent_dict.get("destination") and ent_dict.get("cuisine"):
        recs = mock_restaurant_recommendation(ent_dict["destination"], ent_dict["cuisine"])
        prompt += f"Recommended restaurants in {ent_dict['destination']}: {', '.join(recs)}\n"

    if ent_dict.get("destination"):
        hotels = mock_hotel_suggestions(ent_dict["destination"])
        prompt += f"Hotel options in {ent_dict['destination']}: {', '.join(hotels)}\n"

    return prompt


def generate_response(query, context_docs, model="gpt-3.5-turbo", client=None):
    """
    Generate a response to a travel query using OpenAI and log the interaction.

    Args:
        query (str): User's travel-related question.
        context_docs (list): List of context documents relevant to the query.
        model (str): OpenAI model to use.
        client (OpenAI, optional): Custom OpenAI client for testing or override.

    Returns:
        str: Generated assistant response or error message.
    """
    try:
        # Use default client if none provided
        if client is None:
            client = OpenAI(api_key=api_key)

        # Step 1: Extract entities from query
        extracted_entities = extract_entities(query)

        # Step 2: Enrich context with tool-based information
        tool_context = enrich_prompt_with_tools(query, extracted_entities)

        # Step 3: Add RAG-retrieved documents
        truncated_docs = [doc[:1000] for doc in context_docs]
        context = tool_context + "\n\n" + "\n\n".join(truncated_docs)

        # Step 4: Compose the final prompt
        prompt = f"""You are a travel assistant. Based on the following context, answer the user's travel query.

Context:
{context}

User Query:
{query}

Answer:"""

        # Step 5: Call OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        final_response = response.choices[0].message.content.strip()
        log_interaction(query, context_docs, final_response, extracted_entities)
        return final_response

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        log_interaction(query, context_docs, error_msg)
        return error_msg

def generate_response_without_rag(query, model="gpt-3.5-turbo", client=None):
    """
    Generate a response without using retrieval, purely from the LLM.

    Args:
        query (str): User's travel-related question.
        model (str): OpenAI model to use.
        client (OpenAI, optional): Custom OpenAI client for testing.

    Returns:
        str: Generated assistant response or error message.
    """
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error (no RAG): {e}"

