"""
LlamaIndex Demo - Weather Verification Multi-Agent System

LlamaIndex's agent framework for RAG and tool-using LLMs.

Key features:
- FunctionTool for wrapping Python functions
- OpenAIAgent for tool-calling agents
- AgentRunner for orchestration
- Strong RAG integration (though not used here)

Install:
    pip install llama-index llama-index-agent-openai llama-index-llms-openai

Usage:
    conda activate dory
    python llamaindex_demo.py
"""

import os
import json
import asyncio

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


async def run_llamaindex_demo():
    """Run the LlamaIndex multi-agent demo."""
    try:
        from llama_index.core.tools import FunctionTool
        from llama_index.core.agent import ReActAgent
        from llama_index.llms.openai import OpenAI
    except ImportError:
        print("LlamaIndex not installed. Run: pip install llama-index llama-index-agent-openai llama-index-llms-openai")
        return None

    print(f"\n{'='*60}")
    print("LlamaIndex Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Configure LLM ---
    # LlamaIndex uses OPENAI_API_KEY and OPENAI_BASE_URL from environment
    llm = OpenAI(model=DEFAULT_MODEL, temperature=0)

    # --- Tool Functions ---
    def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert an Australian address to latitude/longitude coordinates using Nominatim.

        Args:
            city: City name (e.g., "Brisbane")
            state: Australian state code (e.g., "QLD")
            postcode: Postcode (e.g., "4000")

        Returns:
            JSON string with latitude, longitude, and display_name
        """
        print(f"  → Geocoding: {city}, {state}, {postcode}")
        result = geocode_address(city, state, postcode)
        print(f"    → {result}")
        return json.dumps(result)

    def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology.

        Args:
            lat: Latitude (e.g., -27.5)
            lon: Longitude (e.g., 153.0)
            date: Date in YYYY-MM-DD format
            state: Australian state code (e.g., "QLD")

        Returns:
            JSON string with thunderstorms and strong_wind observations
        """
        print(f"  → Fetching BOM weather: ({lat}, {lon}) on {date}")
        result = fetch_bom_observations(lat, lon, date, state)
        print(f"    → {result}")
        return json.dumps(result)

    # --- Create Tools ---
    geocode_tool = FunctionTool.from_defaults(
        fn=geocode_location,
        name="geocode_location",
        description="Convert an Australian address to latitude/longitude coordinates."
    )

    weather_tool = FunctionTool.from_defaults(
        fn=get_bom_weather,
        name="get_bom_weather",
        description="Fetch weather observations from Australian Bureau of Meteorology."
    )

    # --- Weather Agent ---
    print("Creating Weather Agent...")

    weather_agent = ReActAgent(
        tools=[geocode_tool, weather_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are a Weather Verification Agent. Your job is to verify severe weather events.

STEPS:
1. Use geocode_location to convert the address to coordinates
2. Use get_bom_weather to fetch weather observations
3. Report your findings in JSON format

Always use your tools - never make up data."""
    )

    # Run weather agent
    print("\nRunning Weather Agent...")
    weather_task = f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
    weather_response = await weather_agent.run(user_msg=weather_task)
    weather_result = str(weather_response)

    print(f"\nWeather Agent Output:\n{weather_result[:500]}...")

    # --- Eligibility Agent ---
    print("\nCreating Eligibility Agent...")

    eligibility_agent = ReActAgent(
        tools=[],  # No tools - pure reasoning
        llm=llm,
        verbose=True,
        system_prompt="""You are a Claims Eligibility Agent. Evaluate weather verification and determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Only ONE "Observed" = POSSIBLE CAT → REVIEW
- Neither "Observed" = NOT CAT → DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}"""
    )

    # Run eligibility agent
    print("\nRunning Eligibility Agent...")
    eligibility_task = f"Evaluate CAT eligibility based on this weather verification:\n\n{weather_result}"
    eligibility_response = await eligibility_agent.run(user_msg=eligibility_task)
    eligibility_result = str(eligibility_response)

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
    asyncio.run(run_llamaindex_demo())
