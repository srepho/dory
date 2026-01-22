"""
CrewAI Demo - Weather Verification Multi-Agent System

Role-based multi-agent collaboration framework.

Key features:
- Agents with role, goal, backstory
- Tasks with descriptions and expected outputs
- Crew orchestration with sequential/hierarchical processes
- Task dependencies via context parameter

Install:
    pip install crewai crewai-tools litellm

Usage:
    conda activate dory
    python crewai_demo.py
"""

import json

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


def run_crewai_demo():
    """Run the CrewAI multi-agent demo."""
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        from crewai.tools import tool
    except ImportError:
        print("CrewAI not installed. Run: pip install crewai crewai-tools litellm")
        return None

    print(f"\n{'='*60}")
    print("CrewAI Demo")
    print(f"{'='*60}")
    print(f"Model: openai/{DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Tools (only for Weather Agent) ---
    @tool("Geocode Location")
    def geocode_tool(city: str, state: str, postcode: str) -> str:
        """Convert address to latitude/longitude coordinates using Nominatim.

        Args:
            city: City name (e.g., "Brisbane")
            state: Australian state code (e.g., "QLD")
            postcode: Postcode (e.g., "4000")
        """
        result = geocode_address(city, state, postcode)
        return json.dumps(result)

    @tool("Fetch BOM Weather")
    def bom_weather_tool(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology.

        Args:
            lat: Latitude (e.g., -27.5)
            lon: Longitude (e.g., 153.0)
            date: Date in YYYY-MM-DD format
            state: Australian state code (e.g., "QLD")
        """
        result = fetch_bom_observations(lat, lon, date, state)
        return json.dumps(result)

    # --- Configure LLM ---
    # CrewAI uses LiteLLM which reads OPENAI_API_KEY and OPENAI_BASE_URL from env
    llm = LLM(model=f"openai/{DEFAULT_MODEL}")

    # --- Agent 1: Weather Verification Specialist ---
    weather_agent = Agent(
        role="Weather Verification Specialist",
        goal="Accurately verify severe weather events by gathering data from official sources",
        backstory="""You are an expert meteorologist specializing in verifying weather
events for insurance claims. You have access to the Australian Bureau of Meteorology
data and always use your tools to gather factual evidence.""",
        tools=[geocode_tool, bom_weather_tool],
        llm=llm,
        verbose=True
    )

    # --- Agent 2: Claims Eligibility Analyst ---
    eligibility_agent = Agent(
        role="Claims Eligibility Analyst",
        goal="Make accurate CAT event eligibility decisions based on weather evidence",
        backstory="""You are a senior claims analyst specializing in catastrophic event
eligibility. You carefully evaluate weather verification reports and apply strict
business rules to determine if claims qualify as CAT events.""",
        tools=[],  # No tools - pure reasoning
        llm=llm,
        verbose=True
    )

    # --- Task 1: Weather Verification ---
    weather_task = Task(
        description=f"""Verify weather conditions for the following claim:
- Location: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE}
- Date: {TEST_DATE}

Steps:
1. Use the Geocode Location tool to convert the address to coordinates
2. Use the Fetch BOM Weather tool to get official weather observations
3. Report findings including: location, coordinates, weather events observed""",
        expected_output="""A structured report containing:
- Location and coordinates
- Weather observations (thunderstorms, strong wind)
- Whether severe weather occurred""",
        agent=weather_agent
    )

    # --- Task 2: Eligibility Decision ---
    eligibility_task = Task(
        description="""Based on the weather verification report, determine CAT eligibility.

Apply these rules:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED

Validate coordinates are in Australia (-44 to -10 lat, 112 to 154 lon).""",
        expected_output="""JSON eligibility decision:
{cat_event_status, eligibility_decision, confidence, reasoning, next_steps}""",
        agent=eligibility_agent,
        context=[weather_task]  # Depends on weather_task output
    )

    # --- Crew: Orchestrate agents ---
    crew = Crew(
        agents=[weather_agent, eligibility_agent],
        tasks=[weather_task, eligibility_task],
        process=Process.sequential,
        verbose=True
    )

    # Run the crew
    print("Running crew...")
    result = crew.kickoff()

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nCrew Output:\n{str(result)[:1500]}")

    return result


if __name__ == "__main__":
    run_crewai_demo()
