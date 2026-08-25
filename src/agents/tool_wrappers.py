# src/agents/tool_wrappers.py

import os
import logging
import requests
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ───────────────────────────────────────
# API Keys and Clients
# ───────────────────────────────────────
GEOAPIFY_KEY = os.getenv("GEOAPIFY_API_KEY")

try:
    import googlemaps
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
except Exception as e:
    gmaps = None
    logging.warning(f"[Google Maps] API not available: {e}")

try:
    from yelpapi import YelpAPI
    yelp = YelpAPI(os.getenv("YELP_API_KEY"))
except Exception as e:
    yelp = None
    logging.warning(f"[Yelp] API not available: {e}")

# ───────────────────────────────────────
# Google Maps Route
# ───────────────────────────────────────
def real_google_maps_route(origin, destination):
    try:
        if not gmaps:
            raise ValueError("Google Maps API not initialized.")
        result = gmaps.directions(origin, destination, mode="driving")
        if result:
            leg = result[0]['legs'][0]
            return f"Driving from {origin} to {destination} takes {leg['duration']['text']} and covers {leg['distance']['text']}."
    except Exception as e:
        logging.warning(f"[Google Maps] Failed to fetch route: {e}")
    return None

# ───────────────────────────────────────
# Yelp or fallback restaurant recommendation
# ───────────────────────────────────────
def real_restaurant_recommendation(city, cuisine=None):
    try:
        if not yelp:
            raise ValueError("Yelp API not initialized.")
        term = f"{cuisine} restaurant" if cuisine else "restaurant"
        results = yelp.search_query(term=term, location=city, limit=3)
        return [biz['name'] for biz in results['businesses']]
    except Exception as e:
        logging.warning(f"[Yelp] Failed to fetch restaurants: {e}")
        return []


@lru_cache(maxsize=128)
def geocode_city(city):
    """Resolve a city through Open-Meteo's public geocoding service."""
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            place = results[0]
            return place["latitude"], place["longitude"], place.get("timezone", "auto")
    except (requests.RequestException, KeyError, ValueError) as error:
        logging.warning("[Geocoding] Failed for %s: %s", city, error)
    return None


def live_weather(city):
    """Return current conditions and a short forecast, or None when unavailable."""
    location = geocode_city(city)
    if not location:
        return None
    latitude, longitude, timezone = location
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,apparent_temperature,precipitation,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 3,
                "timezone": timezone,
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_probability_max", [])
        forecast = "; ".join(
            f"{date}: {low}-{high} C, rain {probability}%"
            for date, low, high, probability in zip(dates, lows, highs, rain)
        )
        return (
            f"Live weather in {city} at {current.get('time', 'the latest update')}: "
            f"{current.get('temperature_2m', 'NA')} C, feels like "
            f"{current.get('apparent_temperature', 'NA')} C, wind "
            f"{current.get('wind_speed_10m', 'NA')} km/h. Forecast: {forecast}. "
            "Source: Open-Meteo."
        )
    except (requests.RequestException, KeyError, ValueError) as error:
        logging.warning("[Open-Meteo] Failed for %s: %s", city, error)
        return None

# ───────────────────────────────────────
# Geoapify Attractions (Free)
# ───────────────────────────────────────
def geoapify_attractions(city, limit=3):
    if not GEOAPIFY_KEY:
        return []
    location = geocode_city(city)
    if not location:
        return []
    try:
        latitude, longitude, _ = location
        url = "https://api.geoapify.com/v2/places"
        params = {
            "categories": "tourism.sightseeing",
            "filter": f"circle:{longitude},{latitude},10000",
            "bias": f"proximity:{longitude},{latitude}",
            "limit": limit,
            "apiKey": GEOAPIFY_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [f["properties"]["name"] for f in data.get("features", []) if "name" in f["properties"]]
    except Exception as e:
        logging.warning(f"[Geoapify] Failed to fetch attractions: {e}")
        return []

# ───────────────────────────────────────
# Hotel Suggestions (Mock for now)
# ───────────────────────────────────────
def mock_hotel_suggestions(city, pet_friendly=True):
    """Retained for compatibility; never present invented hotels as live data."""
    del pet_friendly
    return [f"Hotel options in {city} require a live accommodation provider."]

# ───────────────────────────────────────
# Smart Enrichment: Final Context Generator
# ───────────────────────────────────────
def generate_smart_enrichment(entities):
    lines = []
    dest = entities.get("destination")
    origin = entities.get("origin")
    cuisine = entities.get("cuisine")
    budget = entities.get("budget")
    duration = entities.get("duration")

    if origin and dest:
        route = real_google_maps_route(origin, dest)
        if route:
            lines.append(f"Live route data: {route}")
        else:
            lines.append(f"Live route data from {origin} to {dest} is unavailable.")

    if dest and cuisine:
        recs = real_restaurant_recommendation(dest, cuisine)
        if recs:
            lines.append(f"Live {cuisine} restaurants in {dest}: {', '.join(recs)}")

    if dest:
        weather = live_weather(dest)
        if weather:
            lines.append(weather)
        else:
            lines.append(f"Live weather for {dest} is unavailable.")

        attractions = geoapify_attractions(dest)
        if attractions:
            lines.append(f"Live attraction results in {dest}: {', '.join(attractions)}")
        else:
            lines.append(f"Live attraction data for {dest} is unavailable.")

        rest_fallback = real_restaurant_recommendation(dest)
        if rest_fallback:
            lines.append(f"Live restaurant results: {', '.join(rest_fallback)}")
        else:
            lines.append(f"Live restaurant data for {dest} is unavailable.")

        lines.extend(mock_hotel_suggestions(dest))

    if budget:
        lines.append(f"User's budget is approximately ${budget}.")

    if duration:
        lines.append(f"The user is planning a trip of {duration} days.")

    if not lines:
        lines.append("No specific entities found. Provide a general travel recommendation based on the user's query.")

    return "\n".join(lines)
