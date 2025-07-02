import os
import openai
import json

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_entities_with_openai(user_query: str) -> dict:
    prompt = f"""
Extract the following template from the user's query:

{{
  "origin": "...",
  "destination": "...",
  "type of transport": "...",
  "family": true or false
}}

Leave any field as "NA" or false if not mentioned.

User query: "{user_query}"
Return only a valid JSON object.
"""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured travel information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )

        reply = response.choices[0].message.content.strip()
        print("🧠 Raw OpenAI response:", reply)

        # Clean and parse response
        json_start = reply.find("{")
        json_end = reply.rfind("}") + 1
        json_text = reply[json_start:json_end]

        result = json.loads(json_text)
        return {
            "origin": result.get("origin", "NA"),
            "destination": result.get("destination", "NA"),
            "type of transport": result.get("type of transport", "NA"),
            "family": result.get("family", False)
        }

    except Exception as e:
        print(f"⚠️ Error parsing OpenAI response:\n\n{e}")
        return {
            "origin": "NA",
            "destination": "NA",
            "type of transport": "NA",
            "family": False
        }
