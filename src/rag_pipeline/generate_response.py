import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import log_interaction

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(query, context_docs, model="gpt-3.5-turbo"):
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

        # Log the successful interaction
        log_interaction(query, context_docs, final_response)
        return final_response

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        # Log the failed interaction
        log_interaction(query, context_docs, error_msg)
        return error_msg
