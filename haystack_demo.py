"""
Haystack Demo - Weather Verification Multi-Agent System

deepset's modular NLP/LLM framework with pipeline-based architecture.

Key features:
- Pipeline-based composition
- Custom components with @component decorator
- Flexible document stores and retrievers
- Strong integration with various LLM providers

Install:
    pip install haystack-ai

Usage:
    conda activate dory
    python haystack_demo.py
"""

import os
import json

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


def run_haystack_demo():
    """Run the Haystack multi-agent demo."""
    try:
        from haystack import Pipeline
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage
        from haystack.components.builders import ChatPromptBuilder
    except ImportError:
        print("Haystack not installed. Run: pip install haystack-ai")
        return None

    print(f"\n{'='*60}")
    print("Haystack Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Configure Generator ---
    generator_kwargs = {"model": DEFAULT_MODEL}
    if USE_ZAI:
        # Haystack uses OPENAI_API_KEY and OPENAI_BASE_URL from environment
        pass  # Already set in demo_config

    # --- Tool Definitions (for tool calling) ---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "geocode_location",
                "description": "Convert an Australian address to latitude/longitude coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "state": {"type": "string", "description": "State code"},
                        "postcode": {"type": "string", "description": "Postcode"}
                    },
                    "required": ["city", "state", "postcode"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_bom_weather",
                "description": "Fetch weather from Bureau of Meteorology.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "date": {"type": "string"},
                        "state": {"type": "string"}
                    },
                    "required": ["lat", "lon", "date", "state"]
                }
            }
        }
    ]

    # --- Tool Handler ---
    def execute_tool(name: str, args: dict) -> str:
        """Execute a tool and return the result."""
        print(f"  → Tool call: {name}")
        if name == "geocode_location":
            result = geocode_address(args["city"], args["state"], args["postcode"])
        elif name == "get_bom_weather":
            result = fetch_bom_observations(args["lat"], args["lon"], args["date"], args["state"])
        else:
            result = {"error": f"Unknown tool: {name}"}
        print(f"    → {result}")
        return json.dumps(result)

    # --- Weather Agent Pipeline ---
    print("Building Weather Agent pipeline...")

    weather_generator = OpenAIChatGenerator(
        model=DEFAULT_MODEL,
        generation_kwargs={"tools": tools}
    )

    weather_system = """You are a Weather Verification Agent. Use your tools to:
1. Geocode the location to get coordinates
2. Fetch BOM weather observations for the date
3. Report findings in JSON format

Always use your tools - never make up data."""

    # Run weather agent with tool loop
    print("Running Weather Agent...")
    messages = [
        ChatMessage.from_system(weather_system),
        ChatMessage.from_user(
            f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
        )
    ]

    weather_result = None
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        response = weather_generator.run(messages=messages)
        reply = response["replies"][0]

        # Check for tool calls (Haystack 2.x uses reply.tool_calls attribute)
        if reply.tool_calls:
            # Add assistant message with tool calls
            messages.append(reply)

            # Process each tool call
            for tool_call in reply.tool_calls:
                func_name = tool_call.tool_name
                func_args = tool_call.arguments
                tool_result = execute_tool(func_name, func_args)

                # Add tool result message
                from haystack.dataclasses import ToolCall
                messages.append(ChatMessage.from_tool(
                    tool_result=tool_result,
                    origin=tool_call
                ))
        else:
            # No tool calls - we have the final response
            weather_result = reply.text
            break

    if not weather_result:
        weather_result = "Weather agent did not complete within iteration limit"

    print(f"\nWeather Agent Output:\n{weather_result[:500]}...")

    # --- Eligibility Agent Pipeline ---
    print("\nBuilding Eligibility Agent pipeline...")

    eligibility_generator = OpenAIChatGenerator(model=DEFAULT_MODEL)

    eligibility_system = """You are a Claims Eligibility Agent. Evaluate weather data and determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Only ONE "Observed" = POSSIBLE CAT → REVIEW
- Neither "Observed" = NOT CAT → DENIED

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}"""

    print("Running Eligibility Agent...")
    eligibility_messages = [
        ChatMessage.from_system(eligibility_system),
        ChatMessage.from_user(f"Evaluate CAT eligibility:\n\n{weather_result}")
    ]

    response = eligibility_generator.run(messages=eligibility_messages)
    eligibility_result = response["replies"][0].text

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
    run_haystack_demo()
