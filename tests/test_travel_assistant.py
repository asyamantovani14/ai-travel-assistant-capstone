import os
import re
import json
import tempfile
from datetime import datetime
import sys

# Add src to the path for module resolution (assuming test is run from root)
SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from rag_pipeline.generate_response import generate_response, enrich_prompt_with_tools
from nlp.ner_utils import extract_entities
from utils.logger import log_interaction

# ------------------ FAKE CLIENT SETUP ------------------

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

# ------------------ TESTS ------------------

def test_generate_response_basic():
    query = "Plan a 3-day trip to Rome"
    context = ["Rome is a historic city in Italy with attractions like the Colosseum and Vatican."]
    result = generate_response(query, context, client=FakeClient())
    assert isinstance(result, str)
    assert "Mocked reply" in result

def test_extract_entities_structure():
    result = extract_entities("Plan a trip from Rome to Florence for 3 days")
    assert isinstance(result, dict)
    assert "origin" in result and "destination" in result

def test_enrich_prompt_with_tools_output():
    entities = {"origin": "Rome", "destination": "Florence", "cuisine": "Italian"}
    output = enrich_prompt_with_tools("trip", entities)
    assert "restaurants" in output or "Hotel" in output

def test_generate_response_format():
    query = "Plan a trip to Rome"
    docs = ["Rome is the capital of Italy."]
    response = generate_response(query, docs, client=FakeClient())
    assert response == "Mocked reply"

def test_extract_entities_basic():
    query = "Plan a trip from Paris to Rome with Italian food"
    result = extract_entities(query)
    assert isinstance(result, dict)
    assert "origin" in result or "destination" in result

def test_enrich_prompt_with_partial_entities():
    entities = {"destination": "Rome"}
    result = enrich_prompt_with_tools("Trip", entities)
    assert "Hotel options in Rome" in result

def test_log_interaction_creates_file_and_contains_expected_content():
    query = "Plan a short trip to Rome"
    docs = ["Rome is the capital of Italy and has amazing food."]
    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=None, log_dir=tmp_log_dir)

        log_file = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_file)

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert query in content
            assert response in content
            assert "Generated Response:" in content

def test_log_file_contains_expected_content_with_entities_and_timestamp():
    query = "Plan a 3-day trip to Rome with Italian food"
    docs = ["Rome is the capital of Italy and has amazing food."]
    fake_entities = {
        "destination": "Rome",
        "duration": 3,
        "cuisine": "Italian",
        "origin": None,
        "budget": None
    }

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=fake_entities, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_path)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        timestamp_match = re.search(r"--- Query at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ---", content)
        assert timestamp_match
        datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")

        json_match = re.search(r"Extracted Entities:\n({.*?})\n\nGenerated Response:", content, re.DOTALL)
        assert json_match, "Could not extract JSON entity block"

        entities = json.loads(json_match.group(1))
        assert entities["destination"] == "Rome"
        assert entities["duration"] == 3
        assert entities["cuisine"] == "Italian"
        assert entities["origin"] is None
        assert entities["budget"] is None

def test_multiple_logs_are_appended_correctly():
    query1 = "Trip to Berlin"
    query2 = "Trip to Lisbon"
    docs = ["Sample content"]

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response1 = generate_response(query1, docs, client=FakeClient())
        response2 = generate_response(query2, docs, client=FakeClient())

        log_interaction(query1, docs, response1, log_dir=tmp_log_dir)
        log_interaction(query2, docs, response2, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert content.count("User Query:") == 2
        assert query1 in content and query2 in content
        assert "Mocked reply" in content

def test_generated_log_matches_expected_structure():
    query = "Plan a weekend trip to Paris"
    docs = ["Paris is a romantic city with many attractions."]
    fake_entities = {
        "destination": "Paris",
        "duration": 2,
        "cuisine": "French",
        "origin": "Rome",
        "budget": 400
    }

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=fake_entities, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_path)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check all expected structure markers
        assert "--- Query at" in content
        assert "User Query:\n" in content
        assert "Top Matching Documents:" in content
        assert "Extracted Entities:" in content
        assert '"destination": "Paris"' in content
        assert '"duration": 2' in content
        assert '"origin": "Rome"' in content
        assert "Generated Response:" in content
        assert "Mocked reply" in content


def test_generated_log_matches_golden_file():
    query = "Plan a weekend trip to Paris"
    docs = ["Paris is a romantic city with many attractions."]
    fake_entities = {
        "destination": "Paris",
        "duration": 2,
        "cuisine": "French",
        "origin": "Rome",
        "budget": 400
    }

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=fake_entities, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_path)

        with open(log_path, "r", encoding="utf-8") as f:
            generated = f.read()

        with open("tests/golden_log.txt", "r", encoding="utf-8") as f:
            expected = f.read()

        # Rimuovi il timestamp per confronto stabile
        generated_clean = re.sub(r"--- Query at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ---", "--- Query at <timestamp>", generated)
        expected_clean = re.sub(r"--- Query at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ---", "--- Query at <timestamp>", expected)

        assert generated_clean.strip() == expected_clean.strip(), "Generated log does not match expected golden file."

def test_generated_log_matches_golden_file():
    import os
    import re

    query = "Plan a weekend trip to Paris"
    docs = ["Paris is a romantic city with many attractions."]
    fake_entities = {
        "destination": "Paris",
        "duration": 2,
        "cuisine": "French",
        "origin": "Rome",
        "budget": 400
    }

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        response = generate_response(query, docs, client=FakeClient())
        log_interaction(query, docs, response, extracted_entities=fake_entities, log_dir=tmp_log_dir)

        log_path = os.path.join(tmp_log_dir, "query_log.txt")
        assert os.path.exists(log_path)

        with open(log_path, "r", encoding="utf-8") as f:
            generated = f.read()

        # Normalize timestamp
        generated_clean = re.sub(r"--- Query at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ---", "--- Query at <timestamp>", generated)

        golden_path = "tests/golden_log.txt"
        if not os.path.exists(golden_path):
            print("Golden file not found. Creating it...")
            with open(golden_path, "w", encoding="utf-8") as f:
                f.write(generated_clean)
            assert True, "Golden file created. Review it and rerun the test."
        else:
            with open(golden_path, "r", encoding="utf-8") as f:
                expected = f.read()
            expected_clean = re.sub(r"--- Query at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ---", "--- Query at <timestamp>", expected)

            assert generated_clean.strip() == expected_clean.strip(), "Generated log does not match golden file."


def test_full_multi_source_pipeline():
    from data_loader.multi_loader import load_all_sources
    from rag_pipeline.document_retriever import build_faiss_index, retrieve_top_k
    from rag_pipeline.generate_response import generate_response

    docs = load_all_sources()
    index, _, model = build_faiss_index(docs)
    top_docs = retrieve_top_k("A trip to Tokyo with tea ceremonies", index, docs, model)
    result = generate_response("A trip to Tokyo with tea ceremonies", top_docs)
    
    assert isinstance(result, str)
    assert len(result) > 20
