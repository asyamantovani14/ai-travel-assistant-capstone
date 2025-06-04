import os
import tempfile
import json
import re
import pytest
from datetime import datetime
from rag_pipeline.generate_response import generate_response
from utils.logger import log_interaction

# --- Fake client to simulate OpenAI response and error ---
class FakeCompletions:
    @staticmethod
    def create(*args, **kwargs):
        return type("Resp", (), {
            "choices": [type("msg", (), {"message": type("m", (), {"content": "Mocked assistant reply."})()})()]
        })()

class FakeChat:
    completions = FakeCompletions()

class FakeClient:
    chat = FakeChat()

class BrokenClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                raise Exception("Simulated API failure")

# --- Test 1: Logging correctness with model and entities ---
def test_log_content_has_model_and_json_entities():
    query = "Plan a 3-day trip to Rome with Italian food"
    docs = ["Rome has amazing restaurants and historical sites."]
    model = "gpt-3.5-turbo"
    fake_entities = {
        "destination": "Rome",
        "duration": 3,
        "cuisine": "Italian",
        "origin": None,
        "budget": None
    }

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, model=model, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=fake_entities, model=model, log_dir=tmp_log_dir)

        log_file = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_file)

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check model name
        assert f"Model Used: {model}" in content

        # Extract and validate entities block (JSON)
        match = re.search(r"Extracted Entities:\n(\{.*?\})", content, re.DOTALL)
        assert match, "Missing or malformed entity block"
        try:
            parsed = json.loads(match.group(1))
            assert parsed["destination"] == "Rome"
        except Exception as e:
            pytest.fail(f"Extracted Entities block is not valid JSON: {e}")

        # Validate timestamp format
        ts_match = re.search(r"--- Query at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ---", content)
        assert ts_match, "Missing timestamp"
        datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")

# --- Test 2: Handle extremely long queries ---
def test_generate_response_with_long_query():
    long_query = "Visit Paris and Rome " * 500  # Creates a long prompt
    docs = ["Paris is romantic.", "Rome is ancient."]
    result = generate_response(long_query, docs, client=FakeClient())
    assert isinstance(result, str)
    assert len(result.strip()) > 0

# --- Test 3: Ensure graceful failure when OpenAI API fails ---
def test_generate_response_handles_api_failure():
    query = "Suggest a 2-day trip to Berlin"
    docs = ["Berlin has rich history and nightlife."]
    result = generate_response(query, docs, client=BrokenClient())
    assert result.startswith("Error generating response")
