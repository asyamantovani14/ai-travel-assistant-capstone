from unittest.mock import Mock

from agents import tool_wrappers


def test_missing_providers_never_return_invented_businesses(monkeypatch):
    monkeypatch.setattr(tool_wrappers, "gmaps", None)
    monkeypatch.setattr(tool_wrappers, "yelp", None)

    assert tool_wrappers.real_google_maps_route("Milan", "Rome") is None
    assert tool_wrappers.real_restaurant_recommendation("Rome") == []


def test_live_weather_reports_source_and_update_time(monkeypatch):
    tool_wrappers.geocode_city.cache_clear()
    monkeypatch.setattr(
        tool_wrappers,
        "geocode_city",
        lambda city: (41.9, 12.5, "Europe/Rome"),
    )
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "current": {
            "time": "2026-08-24T12:00",
            "temperature_2m": 29,
            "apparent_temperature": 31,
            "wind_speed_10m": 8,
        },
        "daily": {
            "time": ["2026-08-24"],
            "temperature_2m_max": [32],
            "temperature_2m_min": [21],
            "precipitation_probability_max": [10],
        },
    }
    monkeypatch.setattr(tool_wrappers.requests, "get", lambda *args, **kwargs: response)

    result = tool_wrappers.live_weather("Rome")

    assert "2026-08-24T12:00" in result
    assert "Source: Open-Meteo" in result


def test_enrichment_marks_unavailable_live_data(monkeypatch):
    monkeypatch.setattr(tool_wrappers, "live_weather", lambda city: None)
    monkeypatch.setattr(tool_wrappers, "geoapify_attractions", lambda city: [])
    monkeypatch.setattr(
        tool_wrappers, "real_restaurant_recommendation", lambda city, cuisine=None: []
    )

    result = tool_wrappers.generate_smart_enrichment({"destination": "Rome"})

    assert "Live weather for Rome is unavailable" in result
    assert "Live restaurant data for Rome is unavailable" in result
    assert "Rome Inn" not in result
