"""
Shared utilities for AI Agent Framework demos.

This module provides common functions for:
- Geocoding addresses using Nominatim API
- Fetching weather observations from Australian Bureau of Meteorology (BOM)
- Parsing BOM HTML responses
- Helper functions for weather severity checks

These utilities are used by all framework implementations to ensure consistent behavior.
"""

import asyncio
import re
from typing import Optional
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

# Load environment variables from .env file (for API keys if needed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for shared_utils


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Coordinates:
    """Geographic coordinates."""
    latitude: float
    longitude: float


@dataclass
class WeatherObservations:
    """Weather observations from BOM."""
    thunderstorms: str  # "Observed" or "No reports or observations"
    strong_wind: str    # "Observed" or "No reports or observations"
    raw_html: Optional[str] = None


@dataclass
class GeocodingResult:
    """Result from geocoding an address."""
    success: bool
    coordinates: Optional[Coordinates] = None
    display_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WeatherResult:
    """Result from fetching weather data."""
    success: bool
    observations: Optional[WeatherObservations] = None
    error: Optional[str] = None


# ============================================================================
# Async API Functions
# ============================================================================

async def geocode_address_async(
    city: str,
    state: str,
    postcode: str,
    client: Optional[httpx.AsyncClient] = None
) -> GeocodingResult:
    """
    Convert an Australian address to coordinates using Nominatim API.

    Args:
        city: City name (e.g., "Brisbane")
        state: Australian state code (e.g., "QLD")
        postcode: Postcode (e.g., "4000")
        client: Optional httpx.AsyncClient for connection reuse

    Returns:
        GeocodingResult with coordinates or error message
    """
    query = f"{city}, {state}, {postcode}, Australia"
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "countrycodes": "au",
        "limit": 1
    }
    headers = {
        "User-Agent": "WeatherVerificationAgent/1.0 (github.com/demo)"
    }

    try:
        if client:
            response = await client.get(url, params=params, headers=headers, timeout=10.0)
        else:
            async with httpx.AsyncClient() as new_client:
                response = await new_client.get(url, params=params, headers=headers, timeout=10.0)

        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            return GeocodingResult(
                success=True,
                coordinates=Coordinates(
                    latitude=float(data[0]["lat"]),
                    longitude=float(data[0]["lon"])
                ),
                display_name=data[0].get("display_name")
            )
        else:
            return GeocodingResult(
                success=False,
                error=f"No results found for: {query}"
            )

    except httpx.HTTPStatusError as e:
        return GeocodingResult(success=False, error=f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        return GeocodingResult(success=False, error=f"Request error: {str(e)}")
    except Exception as e:
        return GeocodingResult(success=False, error=f"Unexpected error: {str(e)}")


async def fetch_bom_observations_async(
    lat: float,
    lon: float,
    date: str,
    state: str,
    location: str = "Location",
    client: Optional[httpx.AsyncClient] = None
) -> WeatherResult:
    """
    Fetch weather observations from Australian Bureau of Meteorology.

    Args:
        lat: Latitude (e.g., -27.47)
        lon: Longitude (e.g., 153.03)
        date: Date in YYYY-MM-DD format
        state: Australian state code (e.g., "QLD")
        location: Location name for the query
        client: Optional httpx.AsyncClient for connection reuse

    Returns:
        WeatherResult with observations or error message
    """
    url = "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py"
    params = {
        "lat": round(lat, 1),
        "lon": round(lon, 1),
        "date": date,
        "state": state,
        "location": location,
        "unique_id": f"agent_{abs(hash(f'{lat}{lon}{date}')) % 100000}"
    }

    try:
        if client:
            response = await client.get(url, params=params, timeout=15.0)
        else:
            async with httpx.AsyncClient() as new_client:
                response = await new_client.get(url, params=params, timeout=15.0)

        response.raise_for_status()
        html_content = response.text

        # Parse the HTML response
        observations = parse_bom_html(html_content)
        return WeatherResult(success=True, observations=observations)

    except httpx.HTTPStatusError as e:
        return WeatherResult(success=False, error=f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        return WeatherResult(success=False, error=f"Request error: {str(e)}")
    except Exception as e:
        return WeatherResult(success=False, error=f"Unexpected error: {str(e)}")


def parse_bom_html(html_content: str) -> WeatherObservations:
    """
    Parse BOM HTML response to extract weather observations.

    The BOM returns an HTML page with a table containing weather observations.
    We look for rows with "Thunderstorms" and "Strong Wind" to extract the status.

    Args:
        html_content: Raw HTML from BOM API

    Returns:
        WeatherObservations with thunderstorm and wind status
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Default values
    thunderstorms = "No reports or observations"
    strong_wind = "No reports or observations"

    # Find all table rows
    rows = soup.find_all('tr')

    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 2:
            # Get the text from the first cell (weather type)
            weather_type = cells[0].get_text(strip=True).lower()
            # Get the status from the second cell
            status = cells[1].get_text(strip=True)

            if 'thunderstorm' in weather_type:
                thunderstorms = status if status else "No reports or observations"
            elif 'strong wind' in weather_type or 'wind' in weather_type:
                strong_wind = status if status else "No reports or observations"

    # Also try regex patterns for different HTML structures
    if "Observed" in html_content:
        thunder_match = re.search(r'Thunderstorm[s]?\s*</td>\s*<td[^>]*>\s*(Observed)', html_content, re.IGNORECASE)
        if thunder_match:
            thunderstorms = "Observed"

        wind_match = re.search(r'Strong\s*Wind\s*</td>\s*<td[^>]*>\s*(Observed)', html_content, re.IGNORECASE)
        if wind_match:
            strong_wind = "Observed"

    return WeatherObservations(
        thunderstorms=thunderstorms,
        strong_wind=strong_wind,
        raw_html=html_content[:500] if html_content else None  # Keep first 500 chars for debugging
    )


# ============================================================================
# Sync Wrappers (for frameworks that don't support async)
# ============================================================================

def geocode_address(city: str, state: str, postcode: str) -> dict:
    """
    Synchronous wrapper for geocode_address_async.

    Returns a dict for easy JSON serialization:
    {"latitude": float, "longitude": float} on success
    {"error": str} on failure
    """
    result = asyncio.run(geocode_address_async(city, state, postcode))
    if result.success and result.coordinates:
        return {
            "latitude": result.coordinates.latitude,
            "longitude": result.coordinates.longitude,
            "display_name": result.display_name
        }
    return {"error": result.error or "Unknown error"}


def fetch_bom_observations(lat: float, lon: float, date: str, state: str, location: str = "Location") -> dict:
    """
    Synchronous wrapper for fetch_bom_observations_async.

    Returns a dict for easy JSON serialization:
    {"thunderstorms": str, "strong_wind": str} on success
    {"error": str} on failure
    """
    result = asyncio.run(fetch_bom_observations_async(lat, lon, date, state, location))
    if result.success and result.observations:
        return {
            "thunderstorms": result.observations.thunderstorms,
            "strong_wind": result.observations.strong_wind
        }
    return {"error": result.error or "Unknown error"}


# ============================================================================
# Helper Functions
# ============================================================================

def is_severe_weather(observations: dict) -> bool:
    """
    Determine if weather observations indicate severe weather.

    Severe weather is defined as EITHER thunderstorms OR strong wind observed.

    Args:
        observations: Dict with "thunderstorms" and "strong_wind" keys

    Returns:
        True if either weather type was observed
    """
    thunderstorms = observations.get("thunderstorms", "").lower()
    strong_wind = observations.get("strong_wind", "").lower()

    return "observed" in thunderstorms or "observed" in strong_wind


def is_cat_event(observations: dict) -> str:
    """
    Determine CAT event status based on weather observations.

    Args:
        observations: Dict with "thunderstorms" and "strong_wind" keys

    Returns:
        "CONFIRMED" - Both thunderstorms AND strong wind observed
        "POSSIBLE" - Only one weather type observed
        "NOT_CAT" - Neither observed
    """
    thunderstorms = "observed" in observations.get("thunderstorms", "").lower()
    strong_wind = "observed" in observations.get("strong_wind", "").lower()

    if thunderstorms and strong_wind:
        return "CONFIRMED"
    elif thunderstorms or strong_wind:
        return "POSSIBLE"
    else:
        return "NOT_CAT"


def is_valid_australian_coordinates(lat: float, lon: float) -> bool:
    """
    Check if coordinates are within Australian bounds.

    Australia bounds: lat -44 to -10, lon 112 to 154

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        True if coordinates are within Australia
    """
    return -44 <= lat <= -10 and 112 <= lon <= 154


def format_weather_verification_result(
    location: str,
    coordinates: dict,
    date: str,
    observations: dict
) -> dict:
    """
    Format a standardized weather verification result.

    Args:
        location: Original location string
        coordinates: Dict with latitude/longitude
        date: Date of weather check
        observations: Weather observations dict

    Returns:
        Standardized result dict
    """
    cat_status = is_cat_event(observations)
    severe = is_severe_weather(observations)

    return {
        "location": location,
        "coordinates": coordinates,
        "date": date,
        "weather_events": {
            "thunderstorms": observations.get("thunderstorms", "Unknown"),
            "strong_wind": observations.get("strong_wind", "Unknown")
        },
        "cat_event_status": cat_status,
        "severe_weather_confirmed": severe,
        "reasoning": _generate_weather_reasoning(observations, severe, cat_status)
    }


def _generate_weather_reasoning(observations: dict, severe: bool, cat_status: str) -> str:
    """Generate human-readable reasoning for weather verification."""
    thunderstorms = observations.get("thunderstorms", "Unknown")
    strong_wind = observations.get("strong_wind", "Unknown")

    parts = []
    parts.append(f"BOM records show thunderstorms: {thunderstorms}, strong wind: {strong_wind}.")

    if cat_status == "CONFIRMED":
        parts.append("Both weather types observed, indicating a confirmed CAT event.")
    elif cat_status == "POSSIBLE":
        if "observed" in thunderstorms.lower():
            parts.append("Only thunderstorms observed. Possible CAT event requiring review.")
        else:
            parts.append("Only strong wind observed. Possible CAT event requiring review.")
    else:
        parts.append("Neither thunderstorms nor strong wind observed. Not a CAT event.")

    return " ".join(parts)


# ============================================================================
# Test Function
# ============================================================================

async def test_apis():
    """Test the API functions with Brisbane data."""
    print("Testing Geocoding API...")
    geo_result = await geocode_address_async("Brisbane", "QLD", "4000")
    print(f"Geocoding: {geo_result}")

    if geo_result.success and geo_result.coordinates:
        print("\nTesting BOM API...")
        weather_result = await fetch_bom_observations_async(
            geo_result.coordinates.latitude,
            geo_result.coordinates.longitude,
            "2025-03-07",
            "QLD",
            "Brisbane"
        )
        print(f"Weather: {weather_result}")

        if weather_result.success and weather_result.observations:
            print("\nFormatted Result:")
            result = format_weather_verification_result(
                "Brisbane, QLD, 4000",
                {"latitude": geo_result.coordinates.latitude, "longitude": geo_result.coordinates.longitude},
                "2025-03-07",
                {
                    "thunderstorms": weather_result.observations.thunderstorms,
                    "strong_wind": weather_result.observations.strong_wind
                }
            )
            import json
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(test_apis())
