# %% [markdown]
# # AI Agent Framework Comparison Demo
#
# This notebook demonstrates implementing the same multi-agent system across 5 popular
# AI agent frameworks. Each implementation creates:
#
# 1. **Weather Verification Agent** - Uses tools to geocode locations and fetch BOM weather data
# 2. **Claims Eligibility Agent** - Pure LLM reasoning to determine CAT event eligibility
#
# ## Use Case: Insurance Claims Weather Verification
#
# Verify whether a severe weather (CAT) event occurred at a specific Australian location
# on a given date, then determine if an insurance claim qualifies.
#
# ## Test Case
# - Location: Brisbane, QLD, 4000
# - Date: 2025-03-07
# - Expected: Weather data from BOM, eligibility decision

# %% [markdown]
# ## Setup and Dependencies
#
# Install required packages (uncomment as needed):
# ```bash
# pip install httpx beautifulsoup4 python-dotenv  # Required for all frameworks
# pip install autogen-agentchat autogen-ext[openai]  # AutoGen
# pip install langchain langchain-openai langgraph  # LangGraph
# pip install pydantic-ai  # Pydantic AI
# pip install openai  # OpenAI Agents SDK
# pip install crewai crewai-tools  # CrewAI
# ```
#
# API keys are loaded from `.env` file in the same directory.

# %%
# Common imports and shared utilities

import os
import json
import asyncio
from typing import Optional
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Use z.ai API as primary (cheaper for demos) - OpenAI-compatible endpoint
# Falls back to OpenAI if z.ai not configured
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_API_BASE = os.getenv("ZHIPU_API_BASE", "https://api.z.ai/api/paas/v4")

if ZHIPU_API_KEY:
    # Configure OpenAI-compatible clients to use z.ai
    os.environ["OPENAI_API_KEY"] = ZHIPU_API_KEY
    os.environ["OPENAI_BASE_URL"] = ZHIPU_API_BASE.replace("/chat/completions", "")
    print(f"Using z.ai API (cheaper): {ZHIPU_API_KEY[:12]}...")
    USE_ZAI = True
    # z.ai models: glm-4.5, glm-4.5-air, glm-4.6, glm-4.7
    DEFAULT_MODEL = "glm-4.5-air"  # Fast and cheap for demos
elif os.getenv("OPENAI_API_KEY"):
    print(f"Using OpenAI API: {os.getenv('OPENAI_API_KEY')[:8]}...")
    USE_ZAI = False
    DEFAULT_MODEL = "gpt-4o-mini"
else:
    print("Warning: No API key found. Set ZHIPU_API_KEY or OPENAI_API_KEY in .env")
    USE_ZAI = False
    DEFAULT_MODEL = "gpt-4o-mini"

# Import shared utilities (from shared_utils.py in the same directory)
# These provide geocoding and BOM weather API functions
try:
    from shared_utils import (
        geocode_address,
        fetch_bom_observations,
        geocode_address_async,
        fetch_bom_observations_async,
        is_cat_event,
        is_severe_weather,
        is_valid_australian_coordinates,
        format_weather_verification_result
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    print("Warning: shared_utils.py not found. Using inline implementations.")

# Inline implementations if shared_utils not available
if not SHARED_UTILS_AVAILABLE:
    import httpx
    from bs4 import BeautifulSoup

    def geocode_address(city: str, state: str, postcode: str) -> dict:
        """Convert address to coordinates using Nominatim."""
        query = f"{city}, {state}, {postcode}, Australia"
        response = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "countrycodes": "au"},
            headers={"User-Agent": "WeatherVerificationAgent/1.0"},
            timeout=10.0
        )
        data = response.json()
        if data:
            return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
        return {"error": "Location not found"}

    def fetch_bom_observations(lat: float, lon: float, date: str, state: str, location: str = "Location") -> dict:
        """Fetch weather from BOM."""
        response = httpx.get(
            "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
            params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"},
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
        return {"thunderstorms": thunderstorms, "strong_wind": strong_wind}

    def is_cat_event(obs: dict) -> str:
        thunder = "observed" in obs.get("thunderstorms", "").lower()
        wind = "observed" in obs.get("strong_wind", "").lower()
        if thunder and wind:
            return "CONFIRMED"
        elif thunder or wind:
            return "POSSIBLE"
        return "NOT_CAT"

    def is_valid_australian_coordinates(lat: float, lon: float) -> bool:
        return -44 <= lat <= -10 and 112 <= lon <= 154

# Test data
TEST_CITY = "Brisbane"
TEST_STATE = "QLD"
TEST_POSTCODE = "4000"
TEST_DATE = "2025-03-07"

print(f"Test case: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")

# %% [markdown]
# ---
# ## Framework 1: AutoGen 0.4+
#
# **Microsoft's multi-agent framework with event-driven architecture.**
#
# Key features:
# - `AssistantAgent` for LLM-powered agents
# - `RoundRobinGroupChat` / `SelectorGroupChat` for orchestration
# - Tools defined as async functions
# - Shared conversation context
#
# ```bash
# pip install autogen-agentchat autogen-ext[openai]
# ```

# %%
# AutoGen 0.4+ Implementation

async def run_autogen_demo():
    """Run the AutoGen multi-agent demo."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError:
        print("AutoGen not installed. Run: pip install autogen-agentchat autogen-ext[openai]")
        return None

    # Shared model client (uses z.ai if configured, else OpenAI)
    if USE_ZAI:
        # z.ai requires model_info for non-OpenAI models
        model_client = OpenAIChatCompletionClient(
            model=DEFAULT_MODEL,
            base_url=os.getenv("OPENAI_BASE_URL"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            }
        )
    else:
        model_client = OpenAIChatCompletionClient(model=DEFAULT_MODEL)

    # --- Tools for Weather Agent (use httpx directly to avoid asyncio.run conflict) ---
    async def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert address to latitude/longitude coordinates using Nominatim."""
        import httpx
        query = f"{city}, {state}, {postcode}, Australia"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "countrycodes": "au"},
                headers={"User-Agent": "WeatherVerificationAgent/1.0"},
                timeout=10.0
            )
            data = response.json()
            if data:
                return json.dumps({"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"]), "display_name": data[0].get("display_name")})
            return json.dumps({"error": "Location not found"})

    async def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology."""
        import httpx
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
                params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "autogen"},
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
            return json.dumps({"thunderstorms": thunderstorms, "strong_wind": strong_wind})

    # --- Agent 1: Weather Verification (has tools) ---
    weather_agent = AssistantAgent(
        name="WeatherAgent",
        model_client=model_client,
        tools=[geocode_location, get_bom_weather],
        system_message="""You are a Weather Verification Agent. You MUST complete these steps IN ORDER:

STEP 1: Call geocode_location with city, state, postcode to get coordinates
STEP 2: Call get_bom_weather with the latitude, longitude, date, and state from step 1
STEP 3: Report your complete findings in JSON format:
{
  "location": "city, state, postcode",
  "coordinates": {"latitude": X, "longitude": Y},
  "date": "YYYY-MM-DD",
  "weather_events": {"thunderstorms": "status", "strong_wind": "status"}
}

IMPORTANT: You must call BOTH tools before reporting. Do not stop after just geocoding.""",
    )

    # --- Agent 2: Claims Eligibility (no tools - pure reasoning) ---
    eligibility_agent = AssistantAgent(
        name="ClaimsAgent",
        model_client=model_client,
        tools=[],
        system_message="""You are a Claims Eligibility Agent. You receive weather verification results
and determine CAT event eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning}

After your decision, say TERMINATE to end the conversation.""",
    )

    # --- Orchestration: Direct tool calls + agent reasoning ---
    # Since the model struggles with multi-step tool calling, we call tools directly
    # and have agents do the reasoning

    # Step 1: Call tools directly
    geo_result = await geocode_location(TEST_CITY, TEST_STATE, TEST_POSTCODE)
    geo_data = json.loads(geo_result)

    if "error" not in geo_data:
        weather_result = await get_bom_weather(
            geo_data["latitude"],
            geo_data["longitude"],
            TEST_DATE,
            TEST_STATE
        )
        weather_data = json.loads(weather_result)
    else:
        weather_data = {"error": "Could not geocode location"}

    # Step 2: Have eligibility agent reason about the data
    weather_summary = {
        "location": f"{TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE}",
        "coordinates": {"latitude": geo_data.get("latitude"), "longitude": geo_data.get("longitude")},
        "date": TEST_DATE,
        "weather_events": weather_data
    }

    eligibility_task = f"""Based on this weather verification, determine CAT eligibility:

{json.dumps(weather_summary, indent=2)}

Apply the rules:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT -> APPROVED
- Only ONE "Observed" = POSSIBLE CAT -> REVIEW
- Neither "Observed" = NOT CAT -> DENIED"""

    eligibility_result = await eligibility_agent.run(task=eligibility_task)

    print("\n=== AutoGen Result ===")
    print(f"\nWeather Data (tools called directly):")
    print(f"  Geocoding: {geo_result}")
    print(f"  Weather: {weather_result}")

    print(f"\nEligibility Agent Output:")
    for msg in eligibility_result.messages:
        if hasattr(msg, 'content') and msg.content:
            print(f"  [{msg.source}]: {str(msg.content)[:500]}...")

    return {"weather": weather_summary, "eligibility": eligibility_result}


# Uncomment to run:
# asyncio.run(run_autogen_demo())

# %% [markdown]
# ---
# ## Framework 2: LangGraph
#
# **LangChain's state machine framework for agent orchestration.**
#
# Key features:
# - `StateGraph` for explicit state machine definition
# - `@tool` decorator for tool definitions
# - TypedDict for state schema
# - Conditional edges for complex routing
#
# ```bash
# pip install langchain langchain-openai langgraph
# ```

# %%
# LangGraph Implementation

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

    # --- LLMs (uses z.ai if configured) ---
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
    workflow = StateGraph(ClaimState)
    workflow.add_node("weather_agent", weather_agent_node)
    workflow.add_node("eligibility_agent", eligibility_agent_node)
    workflow.set_entry_point("weather_agent")
    workflow.add_edge("weather_agent", "eligibility_agent")
    workflow.add_edge("eligibility_agent", END)

    app = workflow.compile()

    # Run the workflow
    result = app.invoke({
        "city": TEST_CITY,
        "state": TEST_STATE,
        "postcode": TEST_POSTCODE,
        "date": TEST_DATE,
        "messages": []
    })

    print("\n=== LangGraph Result ===")
    print(f"Coordinates: {result.get('coordinates')}")
    print(f"Weather Data: {result.get('weather_data')}")
    print(f"CAT Status: {result.get('cat_event_status')}")
    print(f"Decision: {result.get('eligibility_decision')}")
    print(f"Reasoning: {result.get('final_reasoning')[:500]}...")

    return result


# Uncomment to run:
# run_langgraph_demo()

# %% [markdown]
# ---
# ## Framework 3: Pydantic AI
#
# **Type-safe agent framework with Pydantic integration.**
#
# Key features:
# - Full Pydantic model support for inputs/outputs
# - `@agent.tool` decorator with RunContext
# - Automatic validation and structured outputs
# - Clean, Pythonic API
#
# ```bash
# pip install pydantic-ai
# ```

# %%
# Pydantic AI Implementation

async def run_pydantic_ai_demo():
    """Run the Pydantic AI multi-agent demo."""
    try:
        from pydantic_ai import Agent, RunContext
        from pydantic import BaseModel
        from dataclasses import dataclass as dc_dataclass
        import httpx
    except ImportError:
        print("Pydantic AI not installed. Run: pip install pydantic-ai")
        return None

    # --- Dependencies ---
    @dc_dataclass
    class AppDependencies:
        http_client: httpx.AsyncClient

    # --- Pydantic Models ---
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
    weather_agent = Agent(
        f'openai:{DEFAULT_MODEL}',  # Uses z.ai model when configured
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
        query = f"{city}, {state}, {postcode}, Australia"
        response = await ctx.deps.http_client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "countrycodes": "au"},
            headers={"User-Agent": "WeatherVerificationAgent/1.0"},
            timeout=10.0
        )
        data = response.json()
        if data:
            return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
        return {"error": "Location not found"}

    @weather_agent.tool
    async def get_bom_weather(ctx: RunContext[AppDependencies], lat: float, lon: float, date: str, state: str) -> dict:
        """Fetch weather observations from BOM."""
        response = await ctx.deps.http_client.get(
            "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
            params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "pydantic"},
            timeout=15.0
        )
        # Simplified parsing - in production use BeautifulSoup
        html = response.text
        thunderstorms = "Observed" if "Observed" in html and "Thunderstorm" in html else "No reports"
        strong_wind = "Observed" if "Observed" in html and "Wind" in html else "No reports"
        return {"thunderstorms": thunderstorms, "strong_wind": strong_wind}

    # --- Agent 2: Claims Eligibility ---
    eligibility_agent = Agent(
        f'openai:{DEFAULT_MODEL}',  # Uses z.ai model when configured
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
    async with httpx.AsyncClient() as client:
        deps = AppDependencies(http_client=client)

        # Step 1: Weather verification
        weather_result = await weather_agent.run(
            f"Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}",
            deps=deps
        )
        weather_output = weather_result.response  # Pydantic AI 1.x uses .response

        # Step 2: Eligibility decision
        eligibility_result = await eligibility_agent.run(
            f"Evaluate CAT eligibility based on:\n{weather_output.model_dump_json(indent=2) if hasattr(weather_output, 'model_dump_json') else weather_output}",
            deps=deps
        )
        eligibility_output = eligibility_result.response

    print("\n=== Pydantic AI Result ===")
    print(f"Weather: {weather_output}")
    print(f"\nEligibility: {eligibility_output}")

    return {"weather": weather_output, "eligibility": eligibility_output}


# Uncomment to run:
# asyncio.run(run_pydantic_ai_demo())

# %% [markdown]
# ---
# ## Framework 4: OpenAI Agents SDK
#
# **OpenAI's official lightweight agent framework.**
#
# Key features:
# - Simple `Agent` class with instructions and tools
# - Explicit handoff via `transfer_to_*` functions
# - `Runner` for orchestration
# - Stateless, lightweight design
#
# ```bash
# pip install openai-agents
# ```

# %%
# OpenAI Agents SDK Implementation

async def run_openai_agents_demo():
    """Run the OpenAI Agents SDK multi-agent demo."""
    try:
        from agents import Agent, Runner
    except ImportError:
        try:
            # Alternative import path
            from openai_agents import Agent, Runner
        except ImportError:
            print("OpenAI Agents SDK not installed. Run: pip install openai-agents")
            return None

    # --- Tools for Weather Agent ---
    def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert address to latitude/longitude coordinates using Nominatim."""
        result = geocode_address(city, state, postcode)
        return json.dumps(result)

    def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology."""
        result = fetch_bom_observations(lat, lon, date, state)
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
    runner = Runner()
    result = await runner.run(
        starting_agent=weather_agent,
        input=f"Verify weather and determine CAT eligibility for: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}"
    )

    print("\n=== OpenAI Agents SDK Result ===")
    # Get final output
    if hasattr(result, 'final_output'):
        print(f"Final output: {result.final_output}")
    elif hasattr(result, 'messages') and result.messages:
        print(f"Final response: {result.messages[-1].content[:500] if hasattr(result.messages[-1], 'content') else result.messages[-1]}...")
    else:
        print(f"Result: {result}")

    return result


# Uncomment to run:
# run_openai_agents_demo()

# %% [markdown]
# ---
# ## Framework 5: CrewAI
#
# **Role-based multi-agent collaboration framework.**
#
# Key features:
# - Agents with role, goal, backstory
# - Tasks with descriptions and expected outputs
# - Crew orchestration with sequential/hierarchical processes
# - Task dependencies via `context` parameter
#
# ```bash
# pip install crewai crewai-tools
# ```

# %%
# CrewAI Implementation

def run_crewai_demo():
    """Run the CrewAI multi-agent demo."""
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import tool
    except ImportError:
        print("CrewAI not installed. Run: pip install crewai crewai-tools")
        return None

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

    # --- Configure LLM for CrewAI (uses z.ai via env vars) ---
    # CrewAI uses LiteLLM which reads OPENAI_API_KEY and OPENAI_BASE_URL
    from crewai import LLM
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
        context=[weather_task]  # Depends on weather_task
    )

    # --- Crew: Orchestrate agents ---
    crew = Crew(
        agents=[weather_agent, eligibility_agent],
        tasks=[weather_task, eligibility_task],
        process=Process.sequential,
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()

    print("\n=== CrewAI Result ===")
    print(f"Result: {str(result)[:500]}...")

    return result


# Uncomment to run:
# run_crewai_demo()

# %% [markdown]
# ---
# ## Summary: Running All Demos
#
# Uncomment and run the demo for your chosen framework:

# %%
# Run a specific demo (uncomment one):

# For AutoGen:
# asyncio.run(run_autogen_demo())

# For LangGraph:
# run_langgraph_demo()

# For Pydantic AI:
# asyncio.run(run_pydantic_ai_demo())

# For OpenAI Agents SDK:
# run_openai_agents_demo()

# For CrewAI:
# run_crewai_demo()

print("Demo file loaded. Uncomment a run command above to execute.")

# %% [markdown]
# ---
# ## Framework Comparison Summary
#
# | Framework | Tool Definition | Multi-Agent Pattern | Best For |
# |-----------|-----------------|---------------------|----------|
# | **AutoGen** | `tools=[async_func]` | RoundRobinGroupChat | Collaborative agents |
# | **LangGraph** | `@tool` decorator | StateGraph with edges | Complex workflows |
# | **Pydantic AI** | `@agent.tool` with RunContext | Manual pipeline | Type-safe outputs |
# | **OpenAI Agents** | Plain functions | Explicit handoffs | Simple orchestration |
# | **CrewAI** | `@tool` decorator | Crew with task deps | Role-based teams |
