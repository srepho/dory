"""
Smolagents Demo - Weather Verification Multi-Agent System

Hugging Face's lightweight agent framework for tool-using LLMs.

Key features:
- Simple @tool decorator for tool definitions
- ToolCallingAgent / CodeAgent for different execution modes
- Sequential orchestration for multi-agent patterns
- Works with LiteLLM for model flexibility

Install:
    pip install smolagents

Usage:
    conda activate dory
    python smolagents_demo.py
"""

import os
import json

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


def run_smolagents_demo():
    """Run the Smolagents multi-agent demo."""
    try:
        from smolagents import tool, ToolCallingAgent, LiteLLMModel
    except ImportError:
        print("Smolagents not installed. Run: pip install smolagents")
        return None

    print(f"\n{'='*60}")
    print("Smolagents Demo")
    print(f"{'='*60}")
    print(f"Model: openai/{DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Configure Model ---
    # Smolagents uses LiteLLM which reads OPENAI_API_KEY and OPENAI_BASE_URL from env
    model = LiteLLMModel(model_id=f"openai/{DEFAULT_MODEL}")

    # --- Tool Definitions ---
    @tool
    def geocode_location(city: str, state: str, postcode: str) -> str:
        """
        Convert an Australian address to latitude/longitude coordinates using Nominatim.

        Args:
            city: City name (e.g., "Brisbane")
            state: Australian state code (e.g., "QLD")
            postcode: Postcode (e.g., "4000")

        Returns:
            JSON string with latitude and longitude
        """
        print(f"  → Geocoding: {city}, {state}, {postcode}")
        result = geocode_address(city, state, postcode)
        print(f"    → {result}")
        return json.dumps(result)

    @tool
    def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """
        Fetch weather observations from Australian Bureau of Meteorology.

        Args:
            lat: Latitude (e.g., -27.5)
            lon: Longitude (e.g., 153.0)
            date: Date in YYYY-MM-DD format
            state: Australian state code (e.g., "QLD")

        Returns:
            JSON string with thunderstorm and wind observations
        """
        print(f"  → Fetching BOM weather: ({lat}, {lon}) on {date}")
        result = fetch_bom_observations(lat, lon, date, state)
        print(f"    → {result}")
        return json.dumps(result)

    # --- Weather Agent ---
    print("Creating Weather Agent...")

    weather_agent = ToolCallingAgent(
        tools=[geocode_location, get_bom_weather],
        model=model,
        instructions="""You are a Weather Verification Agent. Your job is to verify severe weather events.

STEPS:
1. Use geocode_location to convert the address to coordinates
2. Use get_bom_weather to fetch weather observations
3. Report your findings in JSON format

Always use your tools - never make up data."""
    )

    # --- Eligibility Agent (no tools - pure reasoning) ---
    print("Creating Eligibility Agent...")

    eligibility_agent = ToolCallingAgent(
        tools=[],
        model=model,
        instructions="""You are a Claims Eligibility Agent.

Evaluate weather verification data using these rules:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Only ONE "Observed" = POSSIBLE CAT → REVIEW
- Neither "Observed" = NOT CAT → DENIED

Validate coordinates are in Australia (-44 to -10 lat, 112 to 154 lon).

Respond with your eligibility decision in JSON format:
{cat_event_status, eligibility_decision, confidence, reasoning}"""
    )

    # --- Run Sequential Pipeline ---
    print("\nRunning Weather Agent...")
    weather_result = weather_agent.run(
        f"Verify weather for: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
    )

    print(f"\nWeather Agent Output:\n{weather_result}")

    print("\nRunning Eligibility Agent...")
    eligibility_result = eligibility_agent.run(
        f"Evaluate CAT eligibility based on this weather verification:\n\n{weather_result}"
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nWeather Verification:\n{weather_result}")
    print(f"\nEligibility Decision:\n{eligibility_result}")

    return {
        "weather": weather_result,
        "eligibility": eligibility_result
    }


if __name__ == "__main__":
    run_smolagents_demo()
