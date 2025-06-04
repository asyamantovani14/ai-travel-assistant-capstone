import pytest
import sys
import os
import tempfile
from datetime import datetime
import re

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rag_pipeline.generate_response import generate_response, generate_response_without_rag
from utils.logger import log_interaction


# Fake client setup
class FakeCompletions:
    @staticmethod
    def create(*args, **kwargs):
        return type("Resp", (), {
            "choices": [type("msg", (), {"message": type("m", (), {"content": "Mocked reply"})()})()]
        })()

class FakeChat:
    completions = FakeCompletions()

class FakeClient:
    chat = FakeChat()


@pytest.mark.parametrize("query", [
    "Plan a weekend in Rome",
    "Suggest a summer trip for a couple",
    "Explore pet-friendly destinations in Europe",
])
def test_generate_response_without_rag_parametrized(query):
    result = generate_response_without_rag(query, client=FakeClient())
    assert isinstance(result, str)
    assert "Mocked reply" in result


def test_generate_response_without_rag_logs_correctly():
    query = "Suggest budget travel to Madrid"
    response = generate_response_without_rag(query, client=FakeClient())

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        log_interaction(query, [], response, extracted_entities=None, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_path)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check timestamp
        timestamp_match = re.search(r"--- Query at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ---", content)
        assert timestamp_match
        datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")

        # Check log content
        assert query in content
        assert "Mocked reply" in content
        assert "Generated Response" in content


def test_generate_response_with_context_and_entities_logging():
    query = "Plan a food tour from Naples to Florence"
    docs = ["Naples is known for pizza. Florence is famous for art and cuisine."]
    fake_entities = {
        "origin": "Naples",
        "destination": "Florence",
        "cuisine": "Italian",
        "budget": None,
        "duration": None
    }

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=fake_entities, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_path)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "origin" in content
        assert "destination" in content
        assert "Florence" in content
        assert "Generated Response" in content
