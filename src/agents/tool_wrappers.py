# src/agents/tool_wrappers.py

import os
import logging
import requests
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
    return f"Driving from {origin} to {destination} takes approximately 6 hours and covers 500km."

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
        return [f"{cuisine.capitalize()} Delight", f"Authentic {cuisine.capitalize()} Kitchen"] if cuisine else [
            "Gourmet Bistro", "Family Diner", "Healthy Greens"
        ]

# ───────────────────────────────────────
# Geoapify Attractions (Free)
# ───────────────────────────────────────
def geoapify_attractions(city, limit=3):
    try:
        url = "https://api.geoapify.com/v2/places"
        params = {
            "categories": "tourism.sightseeing",
            "filter": f"place:{city}",
            "limit": limit,
            "apiKey": GEOAPIFY_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return [f["properties"]["name"] for f in data.get("features", []) if "name" in f["properties"]]
    except Exception as e:
        logging.warning(f"[Geoapify] Failed to fetch attractions: {e}")
        return []

# ───────────────────────────────────────
# Hotel Suggestions (Mock for now)
# ───────────────────────────────────────
def mock_hotel_suggestions(city, pet_friendly=True):
    hotels = [f"{city} Inn", f"{city} Grand Hotel", f"{city} Stay & Go"]
    return [h + " (Pet Friendly)" for h in hotels] if pet_friendly else hotels

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
        lines.append(real_google_maps_route(origin, dest))

    if dest and cuisine:
        recs = real_restaurant_recommendation(dest, cuisine)
        lines.append(f"Recommended {cuisine} restaurants in {dest}: {', '.join(recs)}")

    if dest:
        attractions = geoapify_attractions(dest)
        if attractions:
            lines.append(f"Top attractions in {dest}: {', '.join(attractions)}")
        else:
            lines.append(f"In {dest}, you may enjoy activities such as city tours, museums, and local food tasting.")

        rest_fallback = real_restaurant_recommendation(dest)
        lines.append(f"Popular restaurants include: {', '.join(rest_fallback)}")

        hotels = mock_hotel_suggestions(dest)
        lines.append(f"Hotel options in {dest}: {', '.join(hotels)}")

    if budget:
        lines.append(f"User's budget is approximately ${budget}.")

    if duration:
        lines.append(f"The user is planning a trip of {duration} days.")

    if not lines:
        lines.append("No specific entities found. Provide a general travel recommendation based on the user's query.")

    return "\n".join(lines)
