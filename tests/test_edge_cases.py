import os
import pytest
from rag_pipeline.generate_response import generate_response

class FakeClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(*args, **kwargs):
                return type("Resp", (), {
                    "choices": [type("msg", (), {"message": type("m", (), {"content": "Mocked response"})()})()]
                })()
        completions = Completions()
    chat = Chat()

# Edge case 1: Empty query
def test_empty_query():
    result = generate_response("", ["Paris is a romantic city."], client=FakeClient())
    assert isinstance(result, str)
    assert "Mocked response" in result

# Edge case 2: All extracted entities None
def test_entities_all_none(monkeypatch):
    from nlp import ner_utils
    monkeypatch.setattr(ner_utils, "extract_entities", lambda q: {
        "destination": None, "origin": None, "duration": None, "cuisine": None, "budget": None
    })
    result = generate_response("Trip", ["General travel tip."], client=FakeClient())
    assert "Mocked response" in result

# Edge case 3: Only origin present
def test_only_origin(monkeypatch):
    from nlp import ner_utils
    monkeypatch.setattr(ner_utils, "extract_entities", lambda q: {
        "destination": None, "origin": "Rome", "duration": None, "cuisine": None, "budget": None
    })
    result = generate_response("Start from Rome", ["Rome is well connected."], client=FakeClient())
    assert "Mocked response" in result

# Multilingual input (Italian)
def test_multilingual_input():
    query = "Pianifica un viaggio di 3 giorni a Roma"
    result = generate_response(query, ["Roma è famosa per la sua storia."], client=FakeClient())
    assert "Mocked response" in result

# Test malformed or incomplete response handling
def test_incomplete_response(monkeypatch):
    class BrokenClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(*args, **kwargs):
                    return type("Resp", (), {"choices": [{}]})()  # Missing message
            completions = Completions()
        chat = Chat()

    result = generate_response("Plan a trip", ["Info"], client=BrokenClient())
    assert "Error generating response" in result
