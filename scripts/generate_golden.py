import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = "Plan a 5-day trip to Japan including Tokyo and Kyoto"

GOLDEN_PATH = "tests/golden_outputs/japan_trip_response.txt"

if not os.path.exists(GOLDEN_PATH):
    print("⏳ Calling ChatGPT API...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": PROMPT}
        ],
        temperature=0.7,
        max_tokens=500
    )
    content = response.choices[0].message.content.strip()

    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Golden file saved to {GOLDEN_PATH}")
else:
    print(f"✅ Golden file already exists: {GOLDEN_PATH}")
