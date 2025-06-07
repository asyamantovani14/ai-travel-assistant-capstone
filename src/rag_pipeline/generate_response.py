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
from utils.csv_logger import save_response_to_csv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def enrich_prompt_with_tools(query, entities):
    prompt = ""
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


def generate_response(query, context_docs, model="gpt-3.5-turbo", client=None, log_dir="logs"):
    try:
        if client is None:
            client = OpenAI(api_key=api_key)

        extracted_entities = extract_entities(query)
        tool_context = enrich_prompt_with_tools(query, extracted_entities)
        truncated_docs = [doc[:1000] for doc in context_docs]
        context = tool_context + "\n\n" + "\n\n".join(truncated_docs)

        final_prompt = f"""You are a travel assistant. Based on the following context, answer the user's travel query.

Context:
{context}

User Query:
{query}

Answer:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        final_response = response.choices[0].message.content.strip()

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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error (no RAG): {e}"
