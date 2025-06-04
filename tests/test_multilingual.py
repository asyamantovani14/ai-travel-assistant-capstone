from rag_pipeline.generate_response import generate_response

class FakeClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(*args, **kwargs):
                return type("Resp", (), {"choices": [type("msg", (), {"message": type("m", (), {"content": "Risposta finta"})()})()]})()
        completions = Completions()
    chat = Chat()

def test_italian_input():
    query = "Pianifica un viaggio di 3 giorni a Roma"
    docs = ["Roma è una città storica con molte attrazioni."]
    response = generate_response(query, docs, client=FakeClient())
    assert "Risposta finta" in response
