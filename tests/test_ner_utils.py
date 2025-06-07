import pytest
from nlp.ner_utils import extract_entities

def test_extract_destination():
    result = extract_entities("Plan a trip to Paris")
    assert result["destination"] == "Paris"

def test_extract_origin_and_destination():
    result = extract_entities("Travel from Rome to Madrid")
    assert result["origin"] == "Rome"
    assert result["destination"] == "Madrid"

def test_extract_cuisine():
    result = extract_entities("Find Italian food in Florence")
    assert result["cuisine"] == "Italian"

def test_extract_budget_digits():
    result = extract_entities("Plan a trip with a budget of $500")
    assert result["budget"] == 500

def test_extract_budget_words():
    result = extract_entities("I can spend around five hundred dollars")
    assert result["budget"] == 500

def test_extract_duration_digits():
    result = extract_entities("Plan a 3-day itinerary to Rome")
    assert result["duration"] == 3

def test_extract_duration_words():
    result = extract_entities("Plan a two-day trip")
    assert result["duration"] == 2

def test_extract_all_fields():
    query = "Travel from New York to Tokyo for 5 days with a budget of 2000 dollars and try Japanese food"
    result = extract_entities(query)
    assert result["origin"] == "New York"
    assert result["destination"] == "Tokyo"
    assert result["duration"] == 5
    assert result["budget"] == 2000
    assert result["cuisine"] == "Japanese"
