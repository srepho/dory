"""
Azure Semantic Kernel Demo - Weather Verification Multi-Agent System

Microsoft's SDK for integrating LLMs with conventional programming.

Key features:
- Kernel-based architecture
- Plugins with @kernel_function decorator
- Semantic and native functions
- Strong Azure integration (also supports OpenAI)

Install:
    pip install semantic-kernel

Usage:
    conda activate dory
    python semantic_kernel_demo.py
"""

import os
import json
import asyncio
import httpx
from bs4 import BeautifulSoup

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)


async def run_semantic_kernel_demo():
    """Run the Semantic Kernel multi-agent demo."""
    try:
        import semantic_kernel as sk
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        from semantic_kernel.functions import kernel_function
        from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
        from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
        from semantic_kernel.contents.chat_history import ChatHistory
    except ImportError:
        print("Semantic Kernel not installed. Run: pip install semantic-kernel")
        return None

    print(f"\n{'='*60}")
    print("Semantic Kernel Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Create Kernel ---
    kernel = sk.Kernel()

    # --- Add OpenAI Chat Service ---
    chat_service = OpenAIChatCompletion(
        service_id="chat",
        ai_model_id=DEFAULT_MODEL,
    )
    kernel.add_service(chat_service)

    # --- Weather Plugin (using httpx directly to avoid async issues) ---
    class WeatherPlugin:
        """Plugin for weather verification tools."""

        @kernel_function(
            name="geocode_location",
            description="Convert an Australian address to latitude/longitude coordinates."
        )
        def geocode_location(self, city: str, state: str, postcode: str) -> str:
            """Geocode an address using Nominatim."""
            print(f"  → Geocoding: {city}, {state}, {postcode}")
            query = f"{city}, {state}, {postcode}, Australia"
            with httpx.Client() as client:
                response = client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "countrycodes": "au"},
                    headers={"User-Agent": "WeatherVerificationAgent/1.0"},
                    timeout=10.0
                )
                data = response.json()
                if data:
                    result = {
                        "latitude": float(data[0]["lat"]),
                        "longitude": float(data[0]["lon"]),
                        "display_name": data[0].get("display_name")
                    }
                else:
                    result = {"error": "Location not found"}
            print(f"    → {result}")
            return json.dumps(result)

        @kernel_function(
            name="get_bom_weather",
            description="Fetch weather observations from Australian Bureau of Meteorology."
        )
        def get_bom_weather(self, lat: float, lon: float, date: str, state: str) -> str:
            """Fetch BOM weather data."""
            print(f"  → Fetching BOM weather: ({lat}, {lon}) on {date}")
            with httpx.Client() as client:
                response = client.get(
                    "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
                    params={
                        "lat": round(lat, 1),
                        "lon": round(lon, 1),
                        "date": date,
                        "state": state,
                        "unique_id": "semantic_kernel"
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
            print(f"    → {result}")
            return json.dumps(result)

    # Add plugin to kernel
    kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

    # --- Weather Agent ---
    print("Running Weather Agent...")

    weather_history = ChatHistory()
    weather_history.add_system_message("""You are a Weather Verification Agent. Your job is to verify severe weather events.

STEPS:
1. Use geocode_location to convert the address to coordinates
2. Use get_bom_weather to fetch weather observations
3. Report your findings in JSON format

Always use your tools - never make up data.""")
    weather_history.add_user_message(
        f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
    )

    # Enable auto function calling
    execution_settings = OpenAIChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )

    weather_response = await chat_service.get_chat_message_content(
        chat_history=weather_history,
        settings=execution_settings,
        kernel=kernel
    )

    weather_result = str(weather_response)
    print(f"\nWeather Agent Output:\n{weather_result[:500]}...")

    # --- Eligibility Agent ---
    print("\nRunning Eligibility Agent...")

    eligibility_history = ChatHistory()
    eligibility_history.add_system_message("""You are a Claims Eligibility Agent. Evaluate weather verification and determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Only ONE "Observed" = POSSIBLE CAT → REVIEW
- Neither "Observed" = NOT CAT → DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}""")
    eligibility_history.add_user_message(
        f"Evaluate CAT eligibility based on:\n\n{weather_result}"
    )

    # No tools for eligibility agent
    eligibility_settings = OpenAIChatPromptExecutionSettings()

    eligibility_response = await chat_service.get_chat_message_content(
        chat_history=eligibility_history,
        settings=eligibility_settings,
        kernel=kernel
    )

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
    asyncio.run(run_semantic_kernel_demo())
