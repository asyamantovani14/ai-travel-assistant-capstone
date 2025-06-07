import os
import csv
import tempfile
import json
from utils.csv_logger import save_response_to_csv

def test_save_response_to_csv_creates_file_and_row():
    query = "Suggest a 3-day trip in Japan"
    response = "You could visit Tokyo, Kyoto and Nara."
    model = "gpt-4"
    entities = {
        "destination": "Japan",
        "duration": 3,
        "origin": None,
        "budget": None,
        "cuisine": None
    }
    prompt = "You are a travel assistant...\nContext: ...\nUser Query: ...\nAnswer:"

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = os.path.join(tmp_dir, "test_log.csv")

        # Run the function
        save_response_to_csv(
            query=query,
            response=response,
            model=model,
            entities=entities,
            prompt=prompt,
            log_file=test_path
        )

        # Check if file exists
        assert os.path.exists(test_path), "CSV file was not created."

        # Read and verify content
        with open(test_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1, "CSV should contain exactly one row."
        row = rows[0]

        assert row["query"] == query
        assert row["response"] == response
        assert row["model"] == model
        assert json.loads(row["entities"])["destination"] == "Japan"
        assert "You are a travel assistant" in row["prompt"]
