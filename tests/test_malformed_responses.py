import pytest
from rag_pipeline.generate_response import generate_response

class MalformedClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(*args, **kwargs):
                return type("Resp", (), {"choices": [{}]})()
        completions = Completions()
    chat = Chat()

def test_malformed_response_handling():
    query = "Suggest a trip"
    docs = ["Europe travel guide"]
    response = generate_response(query, docs, client=MalformedClient())
    assert "Error" in response
