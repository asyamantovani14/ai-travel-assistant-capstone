# src/utils/geo.py

import os
import logging
import requests
from dotenv import load_dotenv

# Load API keys
load_dotenv()
GEOAPIFY_KEY = os.getenv("GEOAPIFY_API_KEY")


def get_attractions(city_name, limit=5):
    """
    Fetches sightseeing places in a given city using Geoapify Places API.

    Args:
        city_name (str): Name of the city.
        limit (int): Number of results to return.

    Returns:
        list: A list of attraction names.
    """
    try:
        url = "https://api.geoapify.com/v2/places"
        params = {
            "categories": "tourism.sightseeing",
            "filter": f"place:{city_name}",
            "limit": limit,
            "apiKey": GEOAPIFY_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        attractions = [f["properties"]["name"] for f in data.get("features", []) if "name" in f["properties"]]
        return attractions
    except Exception as e:
        logging.warning(f"[Geoapify] Failed to fetch attractions for '{city_name}': {e}")
        return []


def is_geoapify_ready():
    return GEOAPIFY_KEY is not None
