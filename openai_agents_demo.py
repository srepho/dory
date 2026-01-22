"""
OpenAI Agents SDK Demo - Weather Verification Multi-Agent System

OpenAI's official lightweight agent framework.

Key features:
- Simple Agent class with instructions and tools
- Explicit handoff via transfer_to_* functions
- Runner for orchestration
- Stateless, lightweight design

NOTE: This framework requires OpenAI API. It does NOT work with alternative
API providers (z.ai, Azure, etc.) due to built-in tracing that requires
OpenAI infrastructure.

Install:
    pip install openai-agents

Usage:
    conda activate dory
    export OPENAI_API_KEY=sk-...  # Must be real OpenAI key
    python openai_agents_demo.py
"""

import os
import json
import asyncio

# Import shared configuration
from demo_config import (
    TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


async def run_openai_agents_demo():
    """Run the OpenAI Agents SDK multi-agent demo."""

    # Check for real OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY_REAL") or os.getenv("OPENAI_API_KEY")
    if not openai_key or not openai_key.startswith("sk-"):
        print("="*60)
        print("OpenAI Agents SDK Demo")
        print("="*60)
        print("\nERROR: This framework requires a real OpenAI API key.")
        print("It does NOT work with alternative providers (z.ai, Azure, etc.)")
        print("\nSet OPENAI_API_KEY_REAL=sk-... in your .env file")
        print("="*60)
        return None

    # Temporarily set the real OpenAI key
    original_key = os.environ.get("OPENAI_API_KEY")
    original_base = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = openai_key
    if "OPENAI_BASE_URL" in os.environ:
        del os.environ["OPENAI_BASE_URL"]

    try:
        from agents import Agent, Runner, function_tool
    except ImportError:
        try:
            from openai_agents import Agent, Runner, function_tool
        except ImportError:
            print("OpenAI Agents SDK not installed. Run: pip install openai-agents")
            # Restore original env
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            if original_base:
                os.environ["OPENAI_BASE_URL"] = original_base
            return None

    print(f"\n{'='*60}")
    print("OpenAI Agents SDK Demo")
    print(f"{'='*60}")
    print(f"Model: gpt-4o-mini (OpenAI only)")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Tools for Weather Agent ---
    @function_tool
    def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert address to latitude/longitude coordinates using Nominatim."""
        print(f"  → Geocoding: {city}, {state}, {postcode}")
        result = geocode_address(city, state, postcode)
        print(f"    → {result}")
        return json.dumps(result)

    @function_tool
    def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology."""
        print(f"  → Fetching BOM weather for ({lat}, {lon}) on {date}")
        result = fetch_bom_observations(lat, lon, date, state)
        print(f"    → {result}")
        return json.dumps(result)

    # --- Agent 2 defined first (for handoff reference) ---
    eligibility_agent = Agent(
        name="Claims Eligibility Agent",
        instructions="""You are a Claims Eligibility Agent. You receive weather verification
results and make the final CAT eligibility decision.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Provide your decision as JSON: {cat_event_status, eligibility_decision, confidence, reasoning}""",
        tools=[]  # No tools - pure reasoning
    )

    # --- Handoff function ---
    def transfer_to_eligibility():
        """Transfer to the Claims Eligibility Agent for final decision."""
        print("  → Handoff to eligibility agent")
        return eligibility_agent

    # --- Agent 1: Weather Verification ---
    weather_agent = Agent(
        name="Weather Verification Agent",
        instructions="""You are a Weather Verification Agent. Your job is to:
1. Use geocode_location to convert the address to coordinates
2. Use get_bom_weather to get weather observations
3. Summarize what weather was observed
4. ALWAYS call transfer_to_eligibility to hand off for the final decision

Do not make eligibility decisions - always hand off after gathering weather data.""",
        tools=[geocode_location, get_bom_weather, transfer_to_eligibility]
    )

    # --- Run the agents ---
    print("Running weather → eligibility pipeline...")
    runner = Runner()
    result = await runner.run(
        starting_agent=weather_agent,
        input=f"Verify weather and determine CAT eligibility for: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if hasattr(result, 'final_output'):
        print(f"\nFinal Output:\n{result.final_output}")
    elif hasattr(result, 'messages') and result.messages:
        last_msg = result.messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        print(f"\nFinal Response:\n{content[:1000]}")
    else:
        print(f"\nResult: {result}")

    # Restore original environment
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key
    if original_base:
        os.environ["OPENAI_BASE_URL"] = original_base

    return result


if __name__ == "__main__":
    asyncio.run(run_openai_agents_demo())
