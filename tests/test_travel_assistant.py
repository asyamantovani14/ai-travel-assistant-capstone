import pytest
import sys
import os

# 👇 Forza l'aggiunta della cartella 'src' al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rag_pipeline.generate_response import generate_response

def test_generate_response_basic():
    query = "Plan a 3-day trip to Rome"
    context = ["Rome is a historic city in Italy with attractions like the Colosseum and Vatican."]
    result = generate_response(query, context)
    assert isinstance(result, str)
    assert "Rome" in result or "Colosseum" in result
