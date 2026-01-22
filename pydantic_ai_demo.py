"""
Pydantic AI Demo - Weather Verification Multi-Agent System

Type-safe agent framework with Pydantic integration.

Key features:
- Full Pydantic model support for inputs/outputs
- @agent.tool decorator with RunContext
- Automatic validation and structured outputs
- Clean, Pythonic API

Install:
    pip install pydantic-ai

Usage:
    conda activate dory
    python pydantic_ai_demo.py
"""

import asyncio

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)


async def run_pydantic_ai_demo():
    """Run the Pydantic AI multi-agent demo."""
    try:
        from pydantic_ai import Agent, RunContext
        from pydantic import BaseModel
        from dataclasses import dataclass
        import httpx
    except ImportError:
        print("Pydantic AI not installed. Run: pip install pydantic-ai")
        return None

    print(f"\n{'='*60}")
    print("Pydantic AI Demo")
    print(f"{'='*60}")
    print(f"Model: openai:{DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Dependencies ---
    @dataclass
    class AppDependencies:
        http_client: httpx.AsyncClient

    # --- Pydantic Models for Structured Output ---
    class WeatherVerificationResult(BaseModel):
        location: str
        latitude: float
        longitude: float
        date: str
        thunderstorms: str
        strong_wind: str
        severe_weather_confirmed: bool
        reasoning: str

    class EligibilityDecision(BaseModel):
        cat_event_status: str  # CONFIRMED, POSSIBLE, NOT_CAT
        eligibility_decision: str  # APPROVED, REVIEW, DENIED
        confidence: str  # HIGH, MEDIUM, LOW
        reasoning: str
        next_steps: list[str]

    # --- Agent 1: Weather Verification ---
    # Pydantic AI 1.x uses output_type and instructions instead of result_type and system_prompt
    weather_agent = Agent(
        f'openai:{DEFAULT_MODEL}',
        deps_type=AppDependencies,
        output_type=WeatherVerificationResult,
        instructions="""You are a Weather Verification Agent. Use your tools to:
1. Geocode the location to coordinates
2. Fetch BOM weather observations
Return a structured verification result. Always use tools - never make up data."""
    )

    @weather_agent.tool
    async def geocode(ctx: RunContext[AppDependencies], city: str, state: str, postcode: str) -> dict:
        """Convert address to latitude/longitude coordinates."""
        print(f"  → Geocoding: {city}, {state}, {postcode}")
        query = f"{city}, {state}, {postcode}, Australia"
        response = await ctx.deps.http_client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "countrycodes": "au"},
            headers={"User-Agent": "WeatherVerificationAgent/1.0"},
            timeout=10.0
        )
        data = response.json()
        if data:
            result = {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
            print(f"    → Found: {result}")
            return result
        return {"error": "Location not found"}

    @weather_agent.tool
    async def get_bom_weather(ctx: RunContext[AppDependencies], lat: float, lon: float, date: str, state: str) -> dict:
        """Fetch weather observations from BOM."""
        print(f"  → Fetching BOM weather for ({lat}, {lon}) on {date}")
        response = await ctx.deps.http_client.get(
            "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
            params={
                "lat": round(lat, 1),
                "lon": round(lon, 1),
                "date": date,
                "state": state,
                "unique_id": "pydantic"
            },
            timeout=15.0
        )
        # Parse HTML response
        html = response.text
        thunderstorms = "Observed" if "Observed" in html and "Thunderstorm" in html else "No reports"
        strong_wind = "Observed" if "Observed" in html and "Wind" in html else "No reports"
        result = {"thunderstorms": thunderstorms, "strong_wind": strong_wind}
        print(f"    → Weather: {result}")
        return result

    # --- Agent 2: Claims Eligibility ---
    eligibility_agent = Agent(
        f'openai:{DEFAULT_MODEL}',
        deps_type=AppDependencies,
        output_type=EligibilityDecision,
        instructions="""You are a Claims Eligibility Agent. Evaluate weather verification and determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED (HIGH confidence)
- Only ONE "Observed" = POSSIBLE CAT -> REVIEW (MEDIUM confidence)
- Neither "Observed" = NOT CAT -> DENIED (HIGH confidence)

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Return a structured eligibility decision."""
    )

    # --- Orchestration: Manual Pipeline ---
    print("Running weather agent...")
    async with httpx.AsyncClient() as client:
        deps = AppDependencies(http_client=client)

        # Step 1: Weather verification
        weather_result = await weather_agent.run(
            f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}",
            deps=deps
        )
        # Pydantic AI 1.44+ uses .output for the Pydantic model
        weather_output = weather_result.output

        print("\nRunning eligibility agent...")
        # Step 2: Eligibility decision
        weather_json = weather_output.model_dump_json(indent=2) if hasattr(weather_output, 'model_dump_json') else str(weather_output)
        eligibility_result = await eligibility_agent.run(
            f"Evaluate CAT eligibility based on:\n{weather_json}",
            deps=deps
        )
        eligibility_output = eligibility_result.output

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nWeather Verification:")
    print(f"  Location: {weather_output.location}")
    print(f"  Coordinates: ({weather_output.latitude}, {weather_output.longitude})")
    print(f"  Date: {weather_output.date}")
    print(f"  Thunderstorms: {weather_output.thunderstorms}")
    print(f"  Strong Wind: {weather_output.strong_wind}")
    print(f"  Severe Weather: {weather_output.severe_weather_confirmed}")
    print(f"  Reasoning: {weather_output.reasoning}")

    print(f"\nEligibility Decision:")
    print(f"  CAT Status: {eligibility_output.cat_event_status}")
    print(f"  Decision: {eligibility_output.eligibility_decision}")
    print(f"  Confidence: {eligibility_output.confidence}")
    print(f"  Reasoning: {eligibility_output.reasoning}")
    print(f"  Next Steps: {eligibility_output.next_steps}")

    return {"weather": weather_output, "eligibility": eligibility_output}


if __name__ == "__main__":
    asyncio.run(run_pydantic_ai_demo())
