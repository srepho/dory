"""
AutoGen 0.4+ Demo - Weather Verification Multi-Agent System

Microsoft's multi-agent framework with event-driven architecture.

Key features:
- AssistantAgent for LLM-powered agents
- RoundRobinGroupChat / SelectorGroupChat for orchestration
- Tools defined as async functions
- Shared conversation context

Install:
    pip install autogen-agentchat autogen-ext[openai]

Usage:
    conda activate dory
    python autogen_demo.py
"""

import os
import json
import asyncio

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)


async def run_autogen_demo():
    """Run the AutoGen multi-agent demo."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError:
        print("AutoGen not installed. Run: pip install autogen-agentchat autogen-ext[openai]")
        return None

    print(f"\n{'='*60}")
    print("AutoGen 0.4+ Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Model Client Configuration ---
    if USE_ZAI:
        # z.ai requires model_info for non-OpenAI models
        model_client = OpenAIChatCompletionClient(
            model=DEFAULT_MODEL,
            base_url=os.getenv("OPENAI_BASE_URL"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            }
        )
    else:
        model_client = OpenAIChatCompletionClient(model=DEFAULT_MODEL)

    # --- Tools for Weather Agent ---
    # Using httpx directly to avoid asyncio.run() conflict in nested event loops
    async def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert address to latitude/longitude coordinates using Nominatim."""
        import httpx
        query = f"{city}, {state}, {postcode}, Australia"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "countrycodes": "au"},
                headers={"User-Agent": "WeatherVerificationAgent/1.0"},
                timeout=10.0
            )
            data = response.json()
            if data:
                return json.dumps({
                    "latitude": float(data[0]["lat"]),
                    "longitude": float(data[0]["lon"]),
                    "display_name": data[0].get("display_name")
                })
            return json.dumps({"error": "Location not found"})

    async def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology."""
        import httpx
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
                params={
                    "lat": round(lat, 1),
                    "lon": round(lon, 1),
                    "date": date,
                    "state": state,
                    "unique_id": "autogen"
                },
                timeout=15.0
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            thunderstorms = "No reports or observations"
            strong_wind = "No reports or observations"
            for row in soup.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    weather_type = cells[0].get_text(strip=True).lower()
                    status = cells[1].get_text(strip=True)
                    if 'thunderstorm' in weather_type:
                        thunderstorms = status or "No reports or observations"
                    elif 'wind' in weather_type:
                        strong_wind = status or "No reports or observations"
            return json.dumps({"thunderstorms": thunderstorms, "strong_wind": strong_wind})

    # --- Agent 1: Weather Verification (has tools) ---
    weather_agent = AssistantAgent(
        name="WeatherAgent",
        model_client=model_client,
        tools=[geocode_location, get_bom_weather],
        system_message="""You are a Weather Verification Agent. You MUST complete these steps IN ORDER:

STEP 1: Call geocode_location with city, state, postcode to get coordinates
STEP 2: Call get_bom_weather with the latitude, longitude, date, and state from step 1
STEP 3: Report your complete findings in JSON format:
{
  "location": "city, state, postcode",
  "coordinates": {"latitude": X, "longitude": Y},
  "date": "YYYY-MM-DD",
  "weather_events": {"thunderstorms": "status", "strong_wind": "status"}
}

IMPORTANT: You must call BOTH tools before reporting. Do not stop after just geocoding.""",
    )

    # --- Agent 2: Claims Eligibility (no tools - pure reasoning) ---
    eligibility_agent = AssistantAgent(
        name="ClaimsAgent",
        model_client=model_client,
        tools=[],
        system_message="""You are a Claims Eligibility Agent. You receive weather verification results
and determine CAT event eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}

After your decision, say TERMINATE to end the conversation.""",
    )

    # --- Orchestration ---
    # Since some models struggle with multi-step tool calling in RoundRobin,
    # we call tools directly and have the eligibility agent do the reasoning

    print("Step 1: Geocoding location...")
    geo_result = await geocode_location(TEST_CITY, TEST_STATE, TEST_POSTCODE)
    geo_data = json.loads(geo_result)
    print(f"  → {geo_result}")

    if "error" not in geo_data:
        print("Step 2: Fetching BOM weather data...")
        weather_result = await get_bom_weather(
            geo_data["latitude"],
            geo_data["longitude"],
            TEST_DATE,
            TEST_STATE
        )
        weather_data = json.loads(weather_result)
        print(f"  → {weather_result}")
    else:
        weather_data = {"error": "Could not geocode location"}

    # Prepare weather summary for eligibility agent
    weather_summary = {
        "location": f"{TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE}",
        "coordinates": {
            "latitude": geo_data.get("latitude"),
            "longitude": geo_data.get("longitude")
        },
        "date": TEST_DATE,
        "weather_events": weather_data
    }

    print("Step 3: Running eligibility agent...")
    eligibility_task = f"""Based on this weather verification, determine CAT eligibility:

{json.dumps(weather_summary, indent=2)}

Apply the rules:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED"""

    eligibility_result = await eligibility_agent.run(task=eligibility_task)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nWeather Data:")
    print(f"  Location: {weather_summary['location']}")
    print(f"  Coordinates: {weather_summary['coordinates']}")
    print(f"  Date: {weather_summary['date']}")
    print(f"  Thunderstorms: {weather_data.get('thunderstorms', 'Unknown')}")
    print(f"  Strong Wind: {weather_data.get('strong_wind', 'Unknown')}")

    print(f"\nEligibility Agent Decision:")
    for msg in eligibility_result.messages:
        if hasattr(msg, 'content') and msg.content:
            print(f"  {str(msg.content)[:800]}")

    return {"weather": weather_summary, "eligibility": eligibility_result}


if __name__ == "__main__":
    asyncio.run(run_autogen_demo())
