"""
Anthropic Claude Demo - Weather Verification Multi-Agent System

Direct Anthropic API with manual orchestration for multi-agent patterns.

Key features:
- Native tool use with Claude models
- Manual orchestration (no built-in multi-agent abstraction)
- Excellent reasoning and instruction following
- Rich tool_use and tool_result message format

Install:
    pip install anthropic

Usage:
    conda activate dory
    export ANTHROPIC_API_KEY=sk-ant-...
    python anthropic_demo.py
"""

import os
import json
import asyncio

# Import shared configuration
from demo_config import (
    TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE, ANTHROPIC_API_KEY
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


async def run_anthropic_demo():
    """Run the Anthropic Claude multi-agent demo."""

    # Check for Anthropic API key
    api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("="*60)
        print("Anthropic Claude Demo")
        print("="*60)
        print("\nERROR: ANTHROPIC_API_KEY not found in .env file")
        print("Get your API key from: https://console.anthropic.com/")
        print("="*60)
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic not installed. Run: pip install anthropic")
        return None

    print(f"\n{'='*60}")
    print("Anthropic Claude Demo")
    print(f"{'='*60}")
    print(f"Model: claude-sonnet-4-20250514")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    client = Anthropic(api_key=api_key)

    # --- Tool Definitions ---
    tools = [
        {
            "name": "geocode_location",
            "description": "Convert an Australian address to latitude/longitude coordinates using Nominatim.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name (e.g., 'Brisbane')"},
                    "state": {"type": "string", "description": "Australian state code (e.g., 'QLD')"},
                    "postcode": {"type": "string", "description": "Postcode (e.g., '4000')"}
                },
                "required": ["city", "state", "postcode"]
            }
        },
        {
            "name": "get_bom_weather",
            "description": "Fetch weather observations from Australian Bureau of Meteorology.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "state": {"type": "string", "description": "Australian state code"}
                },
                "required": ["lat", "lon", "date", "state"]
            }
        }
    ]

    # --- Tool Handler (using httpx directly to avoid asyncio.run conflict) ---
    import httpx
    from bs4 import BeautifulSoup

    async def handle_tool_call(tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result."""
        print(f"  → Tool call: {tool_name}")
        async with httpx.AsyncClient() as http_client:
            if tool_name == "geocode_location":
                query = f"{tool_input['city']}, {tool_input['state']}, {tool_input['postcode']}, Australia"
                response = await http_client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "countrycodes": "au"},
                    headers={"User-Agent": "WeatherVerificationAgent/1.0"},
                    timeout=10.0
                )
                data = response.json()
                if data:
                    result = {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
                else:
                    result = {"error": "Location not found"}
            elif tool_name == "get_bom_weather":
                response = await http_client.get(
                    "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
                    params={
                        "lat": round(tool_input["lat"], 1),
                        "lon": round(tool_input["lon"], 1),
                        "date": tool_input["date"],
                        "state": tool_input["state"],
                        "unique_id": "anthropic"
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
                result = {"thunderstorms": thunderstorms, "strong_wind": strong_wind}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

        print(f"    → {result}")
        return json.dumps(result)

    # --- Agent 1: Weather Verification ---
    print("Running Weather Verification Agent...")

    weather_system = """You are a Weather Verification Agent. Your job is to verify severe weather events for insurance claims.

STEPS:
1. Use geocode_location to convert the address to coordinates
2. Use get_bom_weather to fetch weather observations for the date
3. Summarize your findings in JSON format

Always use your tools - never make up data."""

    weather_messages = [
        {"role": "user", "content": f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"}
    ]

    # Loop until the weather agent completes
    weather_result = None
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=weather_system,
            tools=tools,
            messages=weather_messages
        )

        # Check if we need to handle tool calls
        if response.stop_reason == "tool_use":
            # Add assistant response to messages
            weather_messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Add tool results
            weather_messages.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # Extract final text response
            for block in response.content:
                if hasattr(block, 'text'):
                    weather_result = block.text
                    break
            break

    print(f"\nWeather Agent Output:\n{weather_result[:500]}...")

    # --- Agent 2: Claims Eligibility ---
    print("\nRunning Claims Eligibility Agent...")

    eligibility_system = """You are a Claims Eligibility Agent. You evaluate weather verification reports and determine CAT (catastrophic) event eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT → REVIEW
- Neither "Observed" = NOT CAT → DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}"""

    eligibility_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=eligibility_system,
        messages=[
            {"role": "user", "content": f"Based on this weather verification, determine CAT eligibility:\n\n{weather_result}"}
        ]
    )

    eligibility_result = ""
    for block in eligibility_response.content:
        if hasattr(block, 'text'):
            eligibility_result = block.text
            break

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
    asyncio.run(run_anthropic_demo())
