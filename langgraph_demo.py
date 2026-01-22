"""
LangGraph Demo - Weather Verification Multi-Agent System

LangChain's state machine framework for agent orchestration.

Key features:
- StateGraph for explicit state machine definition
- @tool decorator for tool definitions
- TypedDict for state schema
- Conditional edges for complex routing

Install:
    pip install langchain langchain-openai langgraph

Usage:
    conda activate dory
    python langgraph_demo.py
"""

import os
import json

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


def run_langgraph_demo():
    """Run the LangGraph multi-agent demo."""
    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        from typing import TypedDict, Annotated, Literal
        from operator import add
    except ImportError:
        print("LangGraph not installed. Run: pip install langchain langchain-openai langgraph")
        return None

    print(f"\n{'='*60}")
    print("LangGraph Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- State Definition ---
    class ClaimState(TypedDict):
        city: str
        state: str
        postcode: str
        date: str
        coordinates: dict | None
        weather_data: dict | None
        weather_reasoning: str
        eligibility_decision: str | None
        cat_event_status: str | None
        final_reasoning: str
        messages: Annotated[list, add]

    # --- Tools for Weather Agent ---
    @tool
    def geocode_location_tool(city: str, state: str, postcode: str) -> dict:
        """Convert address to latitude/longitude coordinates using Nominatim."""
        return geocode_address(city, state, postcode)

    @tool
    def fetch_weather_tool(lat: float, lon: float, date: str, state: str) -> dict:
        """Fetch weather data from Bureau of Meteorology."""
        return fetch_bom_observations(lat, lon, date, state)

    # --- LLMs ---
    llm_kwargs = {"model": DEFAULT_MODEL, "temperature": 0}
    if USE_ZAI:
        llm_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")

    weather_llm = ChatOpenAI(**llm_kwargs).bind_tools(
        [geocode_location_tool, fetch_weather_tool]
    )
    eligibility_llm = ChatOpenAI(**llm_kwargs)

    # --- Weather Agent Node ---
    def weather_agent_node(state: ClaimState) -> dict:
        """Weather verification agent with tool calling."""
        print("  → Running weather agent node...")

        system = SystemMessage(content="""You are a Weather Verification Agent. Use your tools to:
1. Geocode the location to get coordinates
2. Fetch BOM weather data for the date
Report findings as JSON with: location, coordinates, weather_events""")

        human = HumanMessage(
            content=f"Check weather for {state['city']}, {state['state']}, {state['postcode']} on {state['date']}"
        )

        messages = [system, human]
        response = weather_llm.invoke(messages)

        # Handle tool calls
        coordinates = None
        weather_data = None

        while response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                print(f"    → Tool call: {tool_call['name']}")
                if tool_call["name"] == "geocode_location_tool":
                    result = geocode_location_tool.invoke(tool_call["args"])
                    coordinates = result
                elif tool_call["name"] == "fetch_weather_tool":
                    result = fetch_weather_tool.invoke(tool_call["args"])
                    weather_data = result
                else:
                    result = {"error": "Unknown tool"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })

            response = weather_llm.invoke(messages)

        return {
            "coordinates": coordinates,
            "weather_data": weather_data,
            "weather_reasoning": response.content,
            "messages": [response]
        }

    # --- Eligibility Agent Node ---
    def eligibility_agent_node(state: ClaimState) -> dict:
        """Eligibility agent - pure LLM reasoning."""
        print("  → Running eligibility agent node...")

        prompt = f"""You are a Claims Eligibility Agent. Based on this weather verification:

Weather Data: {json.dumps(state.get('weather_data', {}))}
Coordinates: {json.dumps(state.get('coordinates', {}))}

Determine CAT eligibility using these rules:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE = POSSIBLE CAT -> REVIEW
- Neither = NOT CAT -> DENIED

Validate coordinates are in Australia (-44 to -10 lat, 112 to 154 lon).

Respond with JSON: {{cat_event_status, eligibility_decision, confidence, reasoning}}"""

        response = eligibility_llm.invoke([HumanMessage(content=prompt)])

        # Parse decision from response
        content = response.content
        decision = "REVIEW"
        cat_status = "POSSIBLE"

        if "APPROVED" in content.upper():
            decision = "APPROVED"
            cat_status = "CONFIRMED"
        elif "DENIED" in content.upper():
            decision = "DENIED"
            cat_status = "NOT_CAT"

        return {
            "eligibility_decision": decision,
            "cat_event_status": cat_status,
            "final_reasoning": content,
            "messages": [response]
        }

    # --- Build the Graph ---
    print("Building state graph...")
    workflow = StateGraph(ClaimState)
    workflow.add_node("weather_agent", weather_agent_node)
    workflow.add_node("eligibility_agent", eligibility_agent_node)
    workflow.set_entry_point("weather_agent")
    workflow.add_edge("weather_agent", "eligibility_agent")
    workflow.add_edge("eligibility_agent", END)

    app = workflow.compile()

    # Run the workflow
    print("Running workflow...")
    result = app.invoke({
        "city": TEST_CITY,
        "state": TEST_STATE,
        "postcode": TEST_POSTCODE,
        "date": TEST_DATE,
        "messages": []
    })

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nCoordinates: {result.get('coordinates')}")
    print(f"Weather Data: {result.get('weather_data')}")
    print(f"CAT Status: {result.get('cat_event_status')}")
    print(f"Decision: {result.get('eligibility_decision')}")
    print(f"\nReasoning:\n{result.get('final_reasoning')[:800]}...")

    return result


if __name__ == "__main__":
    run_langgraph_demo()
