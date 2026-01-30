"""
Microsoft Agent Framework Demo - Weather Verification Multi-Agent System

The new unified framework that consolidates AutoGen and Semantic Kernel.

Key features:
- Unified API across Python, C#, TypeScript
- Built-in OpenTelemetry observability
- Middleware for error handling and retries
- Azure integration

Install:
    pip install agent-framework --pre

Usage:
    conda activate dory
    python ms_agent_framework_demo.py
"""

import os
import json
import asyncio

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE,
    WEATHER_AGENT_SYSTEM_PROMPT, ELIGIBILITY_AGENT_SYSTEM_PROMPT
)

# Import shared utilities for tool implementations
from shared_utils import (
    geocode_address_async, fetch_bom_observations_async
)


async def run_ms_agent_framework_demo():
    """Run the Microsoft Agent Framework multi-agent demo."""

    try:
        from agent_framework import ChatAgent, FunctionTool, tool
        from agent_framework.openai import OpenAIChatClient
    except ImportError:
        print("Microsoft Agent Framework not installed.")
        print("Run: pip install agent-framework --pre")
        return None

    print(f"\n{'='*60}")
    print("Microsoft Agent Framework Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Define Tools using @tool decorator ---
    @tool
    async def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert an Australian address to latitude/longitude coordinates.

        Args:
            city: The city name (e.g., "Brisbane")
            state: The state code (e.g., "QLD")
            postcode: The postcode (e.g., "4000")

        Returns:
            JSON string with latitude and longitude coordinates
        """
        print(f"  → Geocoding: {city}, {state}, {postcode}")
        result = await geocode_address_async(city, state, postcode)
        if result.success and result.coordinates:
            output = {
                "latitude": result.coordinates.latitude,
                "longitude": result.coordinates.longitude,
                "display_name": result.display_name
            }
        else:
            output = {"error": result.error or "Unknown error"}
        print(f"    → {output}")
        return json.dumps(output)

    @tool
    async def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            date: Date in YYYY-MM-DD format
            state: Australian state code (e.g., "QLD")

        Returns:
            JSON string with thunderstorm and strong wind observations
        """
        print(f"  → Fetching BOM weather for ({lat}, {lon}) on {date}")
        result = await fetch_bom_observations_async(lat, lon, date, state)
        if result.success and result.observations:
            output = {
                "thunderstorms": result.observations.thunderstorms,
                "strong_wind": result.observations.strong_wind
            }
        else:
            output = {"error": result.error or "Unknown error"}
        print(f"    → {output}")
        return json.dumps(output)

    # --- Create Chat Client ---
    # Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment (set by demo_config)
    client = OpenAIChatClient(model_id=DEFAULT_MODEL)

    # --- Agent 1: Weather Verification (has tools) ---
    print("Creating Weather Agent...")
    weather_agent = ChatAgent(
        chat_client=client,
        name="WeatherAgent",
        instructions=WEATHER_AGENT_SYSTEM_PROMPT,
        tools=[geocode_location, get_bom_weather]
    )

    # --- Agent 2: Claims Eligibility (no tools - pure reasoning) ---
    print("Creating Eligibility Agent...")
    eligibility_agent = ChatAgent(
        chat_client=client,
        name="ClaimsAgent",
        instructions=ELIGIBILITY_AGENT_SYSTEM_PROMPT
    )

    # --- Run Weather Agent ---
    print("\nStep 1: Running Weather Agent...")
    try:
        weather_task = f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
        weather_result = await weather_agent.run(weather_task)
        weather_output = weather_result.text
        print(f"\nWeather Agent Output:\n{weather_output[:500]}...")
    except Exception as e:
        print(f"Weather agent error: {e}")
        import traceback
        traceback.print_exc()
        return await run_fallback_demo()

    # --- Run Eligibility Agent ---
    print("\nStep 2: Running Eligibility Agent...")
    try:
        eligibility_task = f"Evaluate CAT eligibility based on:\n\n{weather_output}"
        eligibility_result = await eligibility_agent.run(eligibility_task)
        eligibility_output = eligibility_result.text
    except Exception as e:
        print(f"Eligibility agent error: {e}")
        eligibility_output = f"Error: {e}"

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nWeather Verification:\n{weather_output}")
    print(f"\nEligibility Decision:\n{eligibility_output}")

    return {
        "weather": weather_output,
        "eligibility": eligibility_output
    }


async def run_fallback_demo():
    """
    Fallback demo that executes tools directly.
    Used when the Agent Framework API doesn't match expected patterns.
    """
    from shared_utils import is_cat_event

    print("\n--- Running Fallback Demo (Direct Tool Execution) ---")

    # Step 1: Geocode
    print("\nStep 1: Geocoding location...")
    geo_result = await geocode_address_async(TEST_CITY, TEST_STATE, TEST_POSTCODE)

    if not geo_result.success:
        print(f"Geocoding failed: {geo_result.error}")
        return {"error": geo_result.error}

    print(f"  → ({geo_result.coordinates.latitude}, {geo_result.coordinates.longitude})")

    # Step 2: Fetch weather
    print("\nStep 2: Fetching BOM weather data...")
    weather_result = await fetch_bom_observations_async(
        geo_result.coordinates.latitude,
        geo_result.coordinates.longitude,
        TEST_DATE,
        TEST_STATE
    )

    if not weather_result.success:
        print(f"Weather fetch failed: {weather_result.error}")
        return {"error": weather_result.error}

    observations = {
        "thunderstorms": weather_result.observations.thunderstorms,
        "strong_wind": weather_result.observations.strong_wind
    }
    print(f"  → {observations}")

    # Step 3: Determine eligibility
    print("\nStep 3: Evaluating eligibility...")
    cat_status = is_cat_event(observations)

    if cat_status == "CONFIRMED":
        decision = "APPROVED"
        confidence = "HIGH"
    elif cat_status == "POSSIBLE":
        decision = "REVIEW"
        confidence = "MEDIUM"
    else:
        decision = "DENIED"
        confidence = "HIGH"

    eligibility = {
        "cat_event_status": cat_status,
        "eligibility_decision": decision,
        "confidence": confidence,
        "reasoning": f"Thunderstorms: {observations['thunderstorms']}, Strong Wind: {observations['strong_wind']}"
    }

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS (Fallback Mode)")
    print(f"{'='*60}")
    print(f"\nLocation: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE}")
    print(f"Coordinates: ({geo_result.coordinates.latitude}, {geo_result.coordinates.longitude})")
    print(f"Date: {TEST_DATE}")
    print(f"Thunderstorms: {observations['thunderstorms']}")
    print(f"Strong Wind: {observations['strong_wind']}")
    print(f"\nCAT Status: {cat_status}")
    print(f"Decision: {decision}")
    print(f"Confidence: {confidence}")

    return {
        "weather": {
            "location": f"{TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE}",
            "coordinates": {
                "latitude": geo_result.coordinates.latitude,
                "longitude": geo_result.coordinates.longitude
            },
            "observations": observations
        },
        "eligibility": eligibility
    }


if __name__ == "__main__":
    asyncio.run(run_ms_agent_framework_demo())
