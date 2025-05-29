import os
import sys
import pathlib
from dotenv import load_dotenv
from openai import OpenAI

# Ensure src/ is in sys.path for absolute imports
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import log_interaction  # Now works properly

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(query, context_docs, model="gpt-3.5-turbo"):
    """
    Generate a response to a travel query using OpenAI and log the interaction.

    Args:
        query (str): User's travel-related question.
        context_docs (list): List of context documents relevant to the query.
        model (str): OpenAI model to use.

    Returns:
        str: Generated assistant response or error message.
    """
    # Truncate long documents to stay within token limits
    truncated_docs = [doc[:1000] for doc in context_docs]
    context = "\n\n".join(truncated_docs)

    prompt = f"""You are a travel assistant. Based on the following context, answer the user's travel query.

Context:
{context}

User Query:
{query}

Answer:"""

    try:
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
        log_interaction(query, context_docs, final_response)
        return final_response

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        log_interaction(query, context_docs, error_msg)
        return error_msg
