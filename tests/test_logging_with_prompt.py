import os
import tempfile
from datetime import datetime
from utils.logger import log_interaction
from rag_pipeline.generate_response import generate_response

# Fake Client simile a quello usato altrove
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

def test_log_file_contains_prompt_and_model():
    query = "Plan a 2-day trip to Florence with Italian food"
    docs = ["Florence is known for art, food and pet-friendly stays."]
    model = "gpt-4"
    prompt_test = "Prompt test for verification"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Simula la generazione con il client fittizio
        response = generate_response(query, docs, model=model, client=FakeClient())

        # Scrivi manualmente il log usando log_interaction
        log_interaction(
            query=query,
            matched_docs=docs,
            response=response,
            extracted_entities={
                "destination": "Florence",
                "duration": 2,
                "cuisine": "Italian",
                "origin": None,
                "budget": None
            },
            model=model,
            final_prompt=prompt_test,
            log_dir=tmp_dir
        )

        # Verifica che il file sia stato creato correttamente
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(tmp_dir, f"{date_str}_log.txt")

        assert os.path.exists(log_file), f"Expected log file not found: {log_file}"

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert f"Model Used: {model}" in content
        assert "Final Prompt Sent to LLM:" in content
        assert prompt_test in content
