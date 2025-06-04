import os
import difflib
import pytest
import re
from fuzzywuzzy import fuzz
from rag_pipeline.generate_response import generate_response

# Add option for regenerating golden file
def pytest_addoption(parser):
    parser.addoption("--accept-new-golden", action="store_true", help="Accept and overwrite golden files")

# Simple text normalizer
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Test cases
test_cases = [
    {
        "query": "Plan a 5-day trip to Japan including Tokyo and Kyoto",
        "docs": [
            "Tokyo is a bustling city with technology and culture.",
            "Kyoto is famous for its temples, shrines, and traditional tea houses."
        ],
        "golden_filename": "japan_trip_response.txt"
    },
    {
        "query": "Suggest a romantic weekend in Paris",
        "docs": ["Paris is known as the City of Love with the Eiffel Tower, Seine river cruises and charming cafés."],
        "golden_filename": "paris_romantic_response.txt"
    }
]

@pytest.mark.parametrize("case", test_cases)
def test_golden_response_similarity(case, request):
    query = case["query"]
    docs = case["docs"]
    filename = case["golden_filename"]
    golden_path = os.path.join("tests", "golden_outputs", filename)

    actual_response = generate_response(query, docs)
    normalized_actual = normalize_text(actual_response)

    accept = request.config.getoption("--accept-new-golden")

    # Write golden if missing or overwrite requested
    if not os.path.exists(golden_path) or accept:
        os.makedirs(os.path.dirname(golden_path), exist_ok=True)
        with open(golden_path, "w", encoding="utf-8") as f:
            f.write(actual_response)
        if accept:
            print(f"✅ Golden file updated: {golden_path}")
            return
        else:
            pytest.fail(f"Golden file created: {golden_path}. Review and re-run the test.")

    with open(golden_path, "r", encoding="utf-8") as f:
        expected_response = f.read()
        normalized_expected = normalize_text(expected_response)

    similarity = fuzz.token_sort_ratio(normalized_actual, normalized_expected)

    if similarity < 85:
        print(f"\n⚠️  Similarity too low: {similarity}%")
        print("------ DIFF START ------")
        for line in difflib.unified_diff(
            expected_response.splitlines(),
            actual_response.splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm=""
        ):
            print(line)
        print("------ DIFF END ------")

        if accept:
            with open(golden_path, "w", encoding="utf-8") as f:
                f.write(actual_response)
            print("✅ Golden file overwritten due to --accept-new-golden.")
        else:
            pytest.fail(f"Low similarity with golden file ({similarity}%). Use --accept-new-golden to overwrite.")
