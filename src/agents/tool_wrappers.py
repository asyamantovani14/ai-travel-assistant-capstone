# src/agents/tool_wrappers.py

def mock_google_maps_route(origin, destination):
    return f"Driving from {origin} to {destination} takes approximately 6 hours and covers 500km."

def mock_restaurant_recommendation(city, cuisine=None):
    if cuisine:
        return [f"{cuisine.capitalize()} Delight", f"Authentic {cuisine.capitalize()} Kitchen"]
    return ["Gourmet Bistro", "Family Diner", "Healthy Greens"]

def mock_hotel_suggestions(city, pet_friendly=True):
    hotels = [f"{city} Inn", f"{city} Grand Hotel", f"{city} Stay & Go"]
    if pet_friendly:
        return [hotel + " (Pet Friendly)" for hotel in hotels]
    return hotels
