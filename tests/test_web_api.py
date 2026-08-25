from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from web_app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "atlas"}


def test_home_serves_premium_app():
    response = client.get("/")

    assert response.status_code == 200
    assert "Atlas Travel Assistant" in response.text
    assert "itinerary-panel" in response.text


@patch("web_app.get_travel_service")
def test_chat_endpoint_maps_request_to_service(mock_get_service: Mock):
    service = Mock()
    service.plan.return_value = {
        "answer": "A refined plan",
        "sources": [],
        "trip": {"destination": "Rome", "duration": 3},
    }
    mock_get_service.return_value = service

    response = client.post(
        "/api/chat",
        json={
            "message": "Plan three days in Rome",
            "history": [{"role": "user", "content": "I like art"}],
            "filters": {"countries": ["Italy"], "activities": ["Museums"]},
        },
    )

    assert response.status_code == 200
    assert response.json()["trip"]["destination"] == "Rome"
    filters = service.plan.call_args.args[2]
    assert filters.countries == ("italy",)
    assert filters.activities == ("museums",)


def test_chat_endpoint_validates_short_messages():
    response = client.post("/api/chat", json={"message": "x"})

    assert response.status_code == 422
