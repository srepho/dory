"""
Claude Agent SDK Demo - Weather Verification Multi-Agent System

Anthropic's official SDK for building agents with Claude Code capabilities.

Key features:
- Native Claude Code integration (Read, Write, Bash tools)
- In-process MCP server support
- Permission hooks for fine-grained control
- Bundled CLI

Install:
    pip install claude-agent-sdk

Usage:
    conda activate dory
    python claude_agent_sdk_demo.py

Note: This SDK requires Claude Code CLI (bundled) and uses Claude models only.
      Requires ANTHROPIC_API_KEY in environment.
"""

import os
import json
import asyncio
from typing import Any

# Import shared configuration
from demo_config import (
    TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE,
    ANTHROPIC_API_KEY
)

# Import shared utilities for tool implementations
from shared_utils import (
    geocode_address_async,
    fetch_bom_observations_async,
    is_cat_event
)


async def run_claude_agent_sdk_demo():
    """Run the Claude Agent SDK multi-agent demo."""

    # Check for API key
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in environment or .env file")
        print("The Claude Agent SDK requires an Anthropic API key.")
        return None

    try:
        from claude_agent_sdk import (
            query,
            ClaudeSDKClient,
            ClaudeAgentOptions,
            tool,
            create_sdk_mcp_server,
            AssistantMessage,
            TextBlock,
            ResultMessage
        )
    except ImportError:
        print("Claude Agent SDK not installed.")
        print("Run: pip install claude-agent-sdk")
        print("\nNote: This SDK bundles the Claude Code CLI.")
        print("Requires Python 3.10+")
        return None

    print(f"\n{'='*60}")
    print("Claude Agent SDK Demo")
    print(f"{'='*60}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Define Tools as MCP Server ---
    # Claude Agent SDK uses in-process MCP servers with @tool decorator

    @tool("geocode_location", "Convert an Australian address to latitude/longitude coordinates", {
        "city": str,
        "state": str,
        "postcode": str
    })
    async def geocode_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Geocode tool for MCP server."""
        print(f"  → Geocoding: {args['city']}, {args['state']}, {args['postcode']}")
        result = await geocode_address_async(args['city'], args['state'], args['postcode'])

        if result.success and result.coordinates:
            output = {
                "latitude": result.coordinates.latitude,
                "longitude": result.coordinates.longitude,
                "display_name": result.display_name
            }
        else:
            output = {"error": result.error or "Unknown error"}

        print(f"    → {output}")
        return {
            "content": [{"type": "text", "text": json.dumps(output)}]
        }

    @tool("get_bom_weather", "Fetch weather observations from Australian Bureau of Meteorology", {
        "lat": float,
        "lon": float,
        "date": str,
        "state": str
    })
    async def weather_tool(args: dict[str, Any]) -> dict[str, Any]:
        """BOM weather tool for MCP server."""
        print(f"  → Fetching BOM weather for ({args['lat']}, {args['lon']}) on {args['date']}")
        result = await fetch_bom_observations_async(
            args['lat'], args['lon'], args['date'], args['state']
        )

        if result.success and result.observations:
            output = {
                "thunderstorms": result.observations.thunderstorms,
                "strong_wind": result.observations.strong_wind
            }
        else:
            output = {"error": result.error or "Unknown error"}

        print(f"    → {output}")
        return {
            "content": [{"type": "text", "text": json.dumps(output)}]
        }

    # Create MCP server with our tools
    weather_server = create_sdk_mcp_server(
        name="weather-tools",
        version="1.0.0",
        tools=[geocode_tool, weather_tool]
    )

    # --- Configure Agent Options ---
    weather_options = ClaudeAgentOptions(
        mcp_servers={"weather": weather_server},
        allowed_tools=[
            "mcp__weather__geocode_location",
            "mcp__weather__get_bom_weather"
        ],
        system_prompt="""You are a Weather Verification Agent. Your job is to verify severe weather events for insurance claims.

STEPS:
1. Use geocode_location to convert the address to coordinates
2. Use get_bom_weather to fetch BOM observations for the date
3. Report your findings in JSON format:
{
    "location": "city, state, postcode",
    "coordinates": {"latitude": X, "longitude": Y},
    "date": "YYYY-MM-DD",
    "thunderstorms": "status",
    "strong_wind": "status"
}

Always use your tools - never make up data.""",
        max_turns=5
    )

    eligibility_options = ClaudeAgentOptions(
        system_prompt="""You are a Claims Eligibility Agent. Evaluate weather verification and determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}""",
        max_turns=1
    )

    # --- Run Weather Agent ---
    print("Step 1: Running Weather Agent...")
    weather_output = ""

    try:
        async for message in query(
            prompt=f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}",
            options=weather_options
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        weather_output += block.text
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    print(f"Error: {message.result}")

    except Exception as e:
        print(f"Weather agent error: {e}")
        print("\nFalling back to direct tool execution...")
        return await run_fallback_demo()

    print(f"\nWeather Agent Output:\n{weather_output[:500]}...")

    # --- Run Eligibility Agent ---
    print("\nStep 2: Running Eligibility Agent...")
    eligibility_output = ""

    try:
        async for message in query(
            prompt=f"Evaluate CAT eligibility based on:\n\n{weather_output}",
            options=eligibility_options
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        eligibility_output += block.text

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
    Fallback demo that executes tools directly without the SDK.
    Used when Claude Agent SDK isn't available or fails.
    """
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


async def run_with_hooks_demo():
    """
    Advanced demo showing Claude Agent SDK hooks for observability.
    """
    try:
        from claude_agent_sdk import (
            ClaudeSDKClient,
            ClaudeAgentOptions,
            HookMatcher,
            HookContext
        )
    except ImportError:
        print("Claude Agent SDK not available for hooks demo")
        return None

    print(f"\n{'='*60}")
    print("Claude Agent SDK Demo - With Hooks")
    print(f"{'='*60}")

    # --- Define Hooks for Observability ---
    async def log_tool_use(input_data: dict, tool_use_id: str | None, context: HookContext) -> dict:
        """Log all tool usage for observability."""
        tool_name = input_data.get('tool_name', 'unknown')
        tool_input = input_data.get('tool_input', {})
        print(f"  [HOOK] Tool called: {tool_name}")
        print(f"  [HOOK] Input: {json.dumps(tool_input)[:100]}...")
        return {}

    async def log_tool_result(input_data: dict, tool_use_id: str | None, context: HookContext) -> dict:
        """Log tool results."""
        tool_name = input_data.get('tool_name', 'unknown')
        tool_response = input_data.get('tool_response', {})
        print(f"  [HOOK] Tool completed: {tool_name}")
        return {}

    options = ClaudeAgentOptions(
        hooks={
            'PreToolUse': [HookMatcher(hooks=[log_tool_use])],
            'PostToolUse': [HookMatcher(hooks=[log_tool_result])]
        },
        max_turns=3
    )

    print("Hooks configured for PreToolUse and PostToolUse")
    print("(Run the main demo to see hooks in action)")

    return {"hooks_configured": True}


if __name__ == "__main__":
    asyncio.run(run_claude_agent_sdk_demo())
