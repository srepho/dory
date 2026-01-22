# Agent Framework Evaluation Guide

## Purpose

This document provides specifications for building a **consistent example agent** across multiple AI agent frameworks. The goal is to evaluate each framework's developer experience, capabilities, and suitability for a production claims processing system.

All examples should use **public APIs only** to enable easy testing and demonstration.

---

## The Example Agent: Weather Verification Agent

### Use Case
Build an agent that can:
1. Accept a location (city, state, postcode) and date
2. Geocode the location to coordinates
3. Query the Australian Bureau of Meteorology (BOM) for weather observations
4. Determine if a severe weather event (storm/strong wind) occurred
5. Return a structured response with the findings

### Why This Example?
- Uses a real public API (BOM)
- Requires tool/function calling
- Needs basic reasoning (interpret results)
- Simple enough to implement quickly
- Complex enough to show framework differences

---

## The Second Agent: Claims Eligibility Agent

### Use Case
A second agent that:
1. Receives the weather verification results from Agent 1
2. Uses LLM reasoning to evaluate if the claim meets CAT event criteria
3. Considers additional business rules (no tool calls, pure LLM reasoning)
4. Returns a structured eligibility decision

### Why Two Agents?
This demonstrates:
- **Agent-to-agent communication** - How data flows between agents
- **Handoff patterns** - How frameworks manage transitions
- **Pure LLM reasoning** - Not every agent needs tools
- **Orchestration complexity** - Sequential, parallel, or conditional flows
- **State management** - How context is preserved across agents

### Claims Eligibility Agent Behavior

**Input** (from Weather Verification Agent):
```json
{
  "location": "Brisbane, QLD, 4000",
  "coordinates": {"latitude": -27.4698, "longitude": 153.0251},
  "date": "2025-03-07",
  "weather_events": {
    "thunderstorms": "Observed",
    "strong_wind": "Observed"
  },
  "severe_weather_confirmed": true,
  "reasoning": "BOM records show both thunderstorms and strong wind were observed..."
}
```

**Business Rules the Agent Must Apply** (via system prompt):
```
1. CAT Event Confirmation:
   - If BOTH thunderstorms AND strong wind observed â†’ Confirmed CAT
   - If ONLY thunderstorms OR ONLY strong wind â†’ Possible CAT (needs review)
   - If neither observed â†’ Not a CAT event

2. Geographic Validation:
   - Must be within Australia (lat: -44 to -10, lon: 112 to 154)
   - Coordinates must be valid numbers

3. Date Validation:
   - Date must not be in the future
   - Date must be within last 90 days for standard processing

4. Eligibility Decision:
   - APPROVED: Confirmed CAT + valid location + valid date
   - REVIEW: Possible CAT or minor validation concerns
   - DENIED: No CAT event or invalid data
```

**Expected Output**:
```json
{
  "claim_reference": "WV-2025-001",
  "location": "Brisbane, QLD, 4000",
  "incident_date": "2025-03-07",
  "cat_event_status": "CONFIRMED",
  "eligibility_decision": "APPROVED",
  "confidence": "HIGH",
  "reasoning": "Weather verification confirms both thunderstorms and strong wind were observed at the claim location on the incident date. Geographic coordinates are valid for mainland Australia. The incident date is within the acceptable processing window. This claim meets all criteria for CAT event eligibility.",
  "next_steps": ["Proceed to coverage verification", "No excess applies for CAT events"],
  "flags": []
}
```

### System Prompt for Claims Eligibility Agent

```
You are a Claims Eligibility Agent for an insurance company. Your role is to evaluate 
weather verification data and determine if a claim qualifies as a Catastrophic (CAT) event.

You will receive weather verification results from the Weather Verification Agent. 
Based on this data, you must:

1. EVALUATE the weather evidence:
   - "Observed" for BOTH thunderstorms AND strong wind = CONFIRMED CAT
   - "Observed" for ONLY ONE weather type = POSSIBLE CAT (requires human review)
   - "No reports or observations" for both = NOT A CAT EVENT

2. VALIDATE the data:
   - Coordinates must be within Australia (-44 to -10 lat, 112 to 154 lon)
   - Date must not be in the future
   - Date should be within 90 days for standard processing

3. DETERMINE eligibility:
   - APPROVED: Confirmed CAT with valid location and date
   - REVIEW: Possible CAT or minor concerns requiring human review
   - DENIED: No CAT event evidence or invalid/suspicious data

4. PROVIDE reasoning:
   - Explain your decision clearly
   - Note any concerns or flags
   - Suggest next steps

Always respond with a structured JSON decision. Be thorough but concise in your reasoning.
Do not invent or assume data - only use what is provided in the weather verification results.
```

---

## Multi-Agent Orchestration Patterns

Each framework handles multi-agent coordination differently. Evaluate these patterns:

### Pattern 1: Sequential Handoff
```
User Query â†’ Weather Agent â†’ [results] â†’ Eligibility Agent â†’ Final Response
```
Simple linear flow. Agent 1 completes fully before Agent 2 starts.

### Pattern 2: Supervisor/Router
```
                    â”Œâ†’ Weather Agent â”€â”€â”
User Query â†’ Routerâ”€â”¤                  â”œâ†’ Aggregator â†’ Response  
                    â””â†’ Other Agent â”€â”€â”€â”€â”˜
```
A supervisor agent decides which agents to invoke and combines results.

### Pattern 3: Collaborative Chat
```
User Query â†’ Group Chat [Weather Agent, Eligibility Agent, ...]
             Agents discuss and build on each other's responses
```
Agents interact in a shared conversation context.

### Pattern 4: State Machine
```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   GEOCODE   â”‚â”€â”€â”€â”€â†’â”‚FETCH_WEATHERâ”‚â”€â”€â”€â”€â†’â”‚  EVALUATE   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚                   â”‚
                           â–¼                   â–¼
                    [On Error: RETRY]   [If REVIEW: HUMAN]
```
Explicit states with conditional transitions.

---

## Public APIs to Use

### 1. BOM Storms API (Primary)
```
Endpoint: https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py
Method: GET
Parameters:
  - lat: float (e.g., -33.9)
  - lon: float (e.g., 151.2)
  - date: string (YYYY-MM-DD)
  - state: string (e.g., "NSW")
  - location: string (e.g., "Sydney")
  - unique_id: string (any identifier)

Example:
https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py?lat=-33.9&lon=151.2&date=2025-03-07&state=NSW&location=Sydney&unique_id=12345

Response: HTML page with weather observations table
- Parse for "Thunderstorms" and "Strong Wind" rows
- Values: "Observed" or "No reports or observations"
```

### 2. Nominatim Geocoding API (For address â†’ coordinates)
```
Endpoint: https://nominatim.openstreetmap.org/search
Method: GET
Parameters:
  - q: string (address query)
  - format: "json"
  - countrycodes: "au" (limit to Australia)

Example:
https://nominatim.openstreetmap.org/search?q=Sydney,NSW,2000,Australia&format=json&countrycodes=au

Headers Required:
  - User-Agent: "WeatherVerificationAgent/1.0"

Response: JSON array with lat/lon coordinates
```

---

## Expected Agent Behavior

### Input
```json
{
  "city": "Brisbane",
  "state": "QLD", 
  "postcode": "4000",
  "date": "2025-03-07"
}
```

### Expected Output
```json
{
  "location": "Brisbane, QLD, 4000",
  "coordinates": {
    "latitude": -27.4698,
    "longitude": 153.0251
  },
  "date": "2025-03-07",
  "weather_events": {
    "thunderstorms": "Observed",
    "strong_wind": "Observed"
  },
  "severe_weather_confirmed": true,
  "reasoning": "BOM records show both thunderstorms and strong wind were observed at this location on the specified date, confirming severe weather conditions."
}
```

---

## Evaluation Criteria

For each framework, evaluate and document:

| Criterion | Description |
|-----------|-------------|
| **Setup Complexity** | How many dependencies? How much boilerplate? |
| **Tool Definition** | How do you define tools/functions the agent can call? |
| **Orchestration Options** | Sequential, parallel, conditional branching? |
| **Type Safety** | Pydantic models, TypedDict, or loose dicts? |
| **Error Handling** | Built-in retry, fallback mechanisms? |
| **Observability** | Logging, tracing, debugging support? |
| **Multi-Agent Support** | Can agents collaborate? How? |
| **Streaming** | Does it support streaming responses? |
| **Model Flexibility** | Easy to swap between GPT-4, Claude, local models? |
| **Production Readiness** | Async support, rate limiting, caching? |
| **Documentation Quality** | How easy is it to learn? |
| **Community/Maintenance** | Active development? Community size? |

---

## Frameworks to Evaluate

### 1. AutoGen (v0.4+) - Microsoft
**Current framework in Nemo. Recently underwent major rewrite.**

```bash
pip install autogen-agentchat autogen-ext
```

**Key Concepts:**
- `AssistantAgent` - LLM-powered agent
- `ToolAgent` - Agent that can use tools
- `SelectorGroupChat` / `RoundRobinGroupChat` - Multi-agent orchestration
- `Handoff` - Transfer between agents

**Documentation:** https://microsoft.github.io/autogen/

**Implementation Notes:**
- v0.4 is a complete rewrite from v0.2
- Now uses an event-driven architecture
- Better async support
- More modular design

```python
# Example structure for AutoGen 0.4+ with TWO AGENTS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Shared model client
model_client = OpenAIChatCompletionClient(model="gpt-4")

# --- AGENT 1: Weather Verification (has tools) ---
async def geocode_location(city: str, state: str, postcode: str) -> dict:
    """Convert address to latitude/longitude coordinates using Nominatim."""
    # Implementation using shared_utils.geocode_address
    pass

async def fetch_bom_weather(lat: float, lon: float, date: str, state: str) -> dict:
    """Fetch weather observations from Australian Bureau of Meteorology."""
    # Implementation using shared_utils.fetch_bom_observations
    pass

weather_agent = AssistantAgent(
    name="WeatherVerificationAgent",
    model_client=model_client,
    tools=[geocode_location, fetch_bom_weather],
    system_message="""You are a Weather Verification Agent. Your job is to:
1. Geocode the provided location to coordinates
2. Fetch weather observations from BOM for the given date
3. Report whether severe weather (thunderstorms/strong wind) was observed
Always use your tools to gather data. Report findings in a structured format.""",
)

# --- AGENT 2: Claims Eligibility (LLM reasoning only, no tools) ---
eligibility_agent = AssistantAgent(
    name="ClaimsEligibilityAgent",
    model_client=model_client,
    tools=[],  # No tools - pure LLM reasoning
    system_message="""You are a Claims Eligibility Agent. You receive weather verification 
results and determine CAT event eligibility.

RULES:
- BOTH thunderstorms AND strong wind observed = CONFIRMED CAT â†’ APPROVED
- Only ONE weather type observed = POSSIBLE CAT â†’ REVIEW  
- Neither observed = NOT CAT â†’ DENIED

Validate coordinates are in Australia (-44 to -10 lat, 112 to 154 lon).
Validate date is not in future and within 90 days.

Respond with JSON: {eligibility_decision, cat_event_status, confidence, reasoning, next_steps}

After providing your decision, say TERMINATE to end the conversation.""",
)

# --- ORCHESTRATION: RoundRobin (simple sequential) ---
team = RoundRobinGroupChat(
    participants=[weather_agent, eligibility_agent],
    termination_condition=TextMentionTermination("TERMINATE"),
)

# --- ALTERNATIVE: Selector (model chooses next agent) ---
selector_team = SelectorGroupChat(
    participants=[weather_agent, eligibility_agent],
    model_client=model_client,
    termination_condition=TextMentionTermination("TERMINATE"),
)

# Run the team
async def run_claim_verification(city: str, state: str, postcode: str, date: str):
    task = f"Verify weather and determine CAT eligibility for: {city}, {state}, {postcode} on {date}"
    result = await team.run(task=task)
    return result
```

**Multi-Agent Pattern:** RoundRobinGroupChat or SelectorGroupChat
**Handoff Style:** Implicit (shared conversation context)

---

### 2. LangChain + LangGraph
**Most popular framework. LangGraph adds stateful orchestration.**

```bash
pip install langchain langchain-openai langgraph
```

**Key Concepts:**
- `Tool` / `@tool` decorator - Define callable tools
- `ChatPromptTemplate` - Structured prompts
- `AgentExecutor` - Run agents with tools (legacy)
- `StateGraph` (LangGraph) - State machine orchestration
- `nodes` and `edges` - Define workflow

**Documentation:** 
- https://python.langchain.com/docs/
- https://langchain-ai.github.io/langgraph/

**Implementation Notes:**
- LangChain for building blocks
- LangGraph for orchestration (recommended for complex flows)
- Excellent for conditional branching

```python
# Example structure for LangGraph with TWO AGENTS
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Literal
from operator import add

# --- STATE DEFINITION ---
class ClaimState(TypedDict):
    # Input
    city: str
    state: str
    postcode: str
    date: str
    # After geocoding
    coordinates: dict | None
    # After weather check
    weather_data: dict | None
    weather_reasoning: str
    # After eligibility check
    eligibility_decision: str | None
    cat_event_status: str | None
    final_reasoning: str
    # Message history
    messages: Annotated[list, add]

# --- TOOLS FOR WEATHER AGENT ---
@tool
def geocode_location(city: str, state: str, postcode: str) -> dict:
    """Convert address to latitude/longitude coordinates using Nominatim."""
    # Implementation using shared_utils
    pass

@tool  
def fetch_bom_weather(lat: float, lon: float, date: str, state: str) -> dict:
    """Fetch weather data from Bureau of Meteorology."""
    # Implementation using shared_utils
    pass

# --- AGENT 1: Weather Verification Node ---
weather_llm = ChatOpenAI(model="gpt-4").bind_tools([geocode_location, fetch_bom_weather])

def weather_agent_node(state: ClaimState) -> ClaimState:
    """Weather verification agent with tool access."""
    system = SystemMessage(content="You verify weather events. Use tools to geocode and fetch BOM data.")
    human = HumanMessage(content=f"Check weather for {state['city']}, {state['state']} on {state['date']}")
    
    # Tool calling loop
    response = weather_llm.invoke([system, human])
    # ... handle tool calls, execute tools, get final response
    
    return {
        "coordinates": {"latitude": ..., "longitude": ...},
        "weather_data": {...},
        "weather_reasoning": response.content,
        "messages": [response]
    }

# --- AGENT 2: Eligibility Agent Node (no tools, pure reasoning) ---
eligibility_llm = ChatOpenAI(model="gpt-4")

ELIGIBILITY_PROMPT = """You are a Claims Eligibility Agent. Based on the weather verification:

Weather Data: {weather_data}
Weather Reasoning: {weather_reasoning}

Determine CAT eligibility:
- BOTH thunderstorms AND strong wind = CONFIRMED â†’ APPROVED
- Only ONE = POSSIBLE â†’ REVIEW
- Neither = NOT CAT â†’ DENIED

Respond with your decision and detailed reasoning."""

def eligibility_agent_node(state: ClaimState) -> ClaimState:
    """Eligibility agent - pure LLM reasoning, no tools."""
    prompt = ELIGIBILITY_PROMPT.format(
        weather_data=state["weather_data"],
        weather_reasoning=state["weather_reasoning"]
    )
    
    response = eligibility_llm.invoke([HumanMessage(content=prompt)])
    
    # Parse response for decision
    decision = "APPROVED" if "APPROVED" in response.content else "REVIEW" if "REVIEW" in response.content else "DENIED"
    
    return {
        "eligibility_decision": decision,
        "cat_event_status": "CONFIRMED" if decision == "APPROVED" else "POSSIBLE" if decision == "REVIEW" else "NOT_CAT",
        "final_reasoning": response.content,
        "messages": [response]
    }

# --- CONDITIONAL ROUTING (optional) ---
def should_continue(state: ClaimState) -> Literal["eligibility", "human_review", "end"]:
    """Route based on weather results."""
    if state.get("weather_data", {}).get("error"):
        return "human_review"
    return "eligibility"

# --- BUILD THE GRAPH ---
workflow = StateGraph(ClaimState)

# Add nodes
workflow.add_node("weather_agent", weather_agent_node)
workflow.add_node("eligibility_agent", eligibility_agent_node)

# Add edges (sequential flow)
workflow.set_entry_point("weather_agent")
workflow.add_edge("weather_agent", "eligibility_agent")
workflow.add_edge("eligibility_agent", END)

# ALTERNATIVE: Conditional routing
# workflow.add_conditional_edges("weather_agent", should_continue, {
#     "eligibility": "eligibility_agent",
#     "human_review": "human_review_node",
#     "end": END
# })

# Compile
app = workflow.compile()

# Run
async def run_claim_verification(city: str, state: str, postcode: str, date: str):
    result = await app.ainvoke({
        "city": city, "state": state, "postcode": postcode, "date": date,
        "messages": []
    })
    return result
```

**Multi-Agent Pattern:** StateGraph with nodes per agent
**Handoff Style:** Explicit state passing between nodes
**Key Advantage:** Conditional routing with `add_conditional_edges`

---

### 3. Pydantic AI
**New framework focused on type safety and validation.**

```bash
pip install pydantic-ai
```

**Key Concepts:**
- `Agent` - Core agent class
- `Tool` - Type-safe tool definitions
- `RunContext` - Dependency injection for tools
- Pydantic models for inputs/outputs
- Built-in retry and validation

**Documentation:** https://ai.pydantic.dev/

**Implementation Notes:**
- Excellent type safety with Pydantic v2
- Clean, Pythonic API
- Good for structured outputs
- Newer, smaller community

```python
# Example structure for Pydantic AI with TWO AGENTS
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dataclasses import dataclass
import httpx

# --- SHARED DEPENDENCIES ---
@dataclass
class AppDependencies:
    http_client: httpx.AsyncClient

# --- PYDANTIC MODELS FOR TYPE SAFETY ---
class WeatherRequest(BaseModel):
    city: str
    state: str
    postcode: str
    date: str

class Coordinates(BaseModel):
    latitude: float
    longitude: float

class WeatherData(BaseModel):
    thunderstorms: str
    strong_wind: str

class WeatherVerificationResult(BaseModel):
    location: str
    coordinates: Coordinates
    date: str
    weather_events: WeatherData
    severe_weather_confirmed: bool
    reasoning: str

class EligibilityDecision(BaseModel):
    claim_reference: str
    location: str
    incident_date: str
    cat_event_status: str  # CONFIRMED, POSSIBLE, NOT_CAT
    eligibility_decision: str  # APPROVED, REVIEW, DENIED
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    next_steps: list[str]
    flags: list[str]

# --- AGENT 1: Weather Verification (with tools) ---
weather_agent = Agent(
    'openai:gpt-4',
    deps_type=AppDependencies,
    result_type=WeatherVerificationResult,
    system_prompt="""You are a Weather Verification Agent. Use your tools to:
1. Geocode the location to coordinates
2. Fetch BOM weather observations
3. Return a structured verification result

Always use tools to get real data. Do not make up coordinates or weather data."""
)

@weather_agent.tool
async def geocode_location(ctx: RunContext[AppDependencies], city: str, state: str, postcode: str) -> dict:
    """Convert address to latitude/longitude coordinates using Nominatim."""
    query = f"{city}, {state}, {postcode}, Australia"
    response = await ctx.deps.http_client.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "countrycodes": "au"},
        headers={"User-Agent": "WeatherVerificationAgent/1.0"}
    )
    data = response.json()
    if data:
        return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
    return {"error": "Location not found"}

@weather_agent.tool
async def fetch_bom_weather(ctx: RunContext[AppDependencies], lat: float, lon: float, date: str, state: str) -> dict:
    """Fetch weather observations from Australian Bureau of Meteorology."""
    response = await ctx.deps.http_client.get(
        "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
        params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
    )
    # Parse HTML response...
    return {"thunderstorms": "Observed", "strong_wind": "Observed"}  # Simplified

# --- AGENT 2: Claims Eligibility (no tools, LLM reasoning) ---
eligibility_agent = Agent(
    'openai:gpt-4',
    deps_type=AppDependencies,
    result_type=EligibilityDecision,
    system_prompt="""You are a Claims Eligibility Agent. You receive weather verification results 
and determine CAT event eligibility based on these rules:

EVALUATION:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT
- Only ONE weather type "Observed" = POSSIBLE CAT
- Neither "Observed" = NOT A CAT EVENT

DECISION:
- CONFIRMED CAT + valid data = APPROVED (HIGH confidence)
- POSSIBLE CAT = REVIEW (MEDIUM confidence)  
- NOT CAT = DENIED (HIGH confidence)

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
- Date must not be in future

Always provide clear reasoning and next steps."""
)
# Note: No @eligibility_agent.tool decorators - this agent uses pure LLM reasoning

# --- ORCHESTRATION: Manual Pipeline ---
async def run_claim_verification(request: WeatherRequest) -> EligibilityDecision:
    """Run both agents in sequence."""
    async with httpx.AsyncClient() as client:
        deps = AppDependencies(http_client=client)
        
        # Step 1: Weather verification (with tools)
        weather_result = await weather_agent.run(
            f"Verify weather for {request.city}, {request.state}, {request.postcode} on {request.date}",
            deps=deps
        )
        
        # Step 2: Eligibility decision (LLM reasoning on weather results)
        eligibility_result = await eligibility_agent.run(
            f"Evaluate CAT eligibility based on this weather verification:\n{weather_result.data.model_dump_json(indent=2)}",
            deps=deps
        )
        
        return eligibility_result.data

# Usage
async def main():
    request = WeatherRequest(city="Brisbane", state="QLD", postcode="4000", date="2025-03-07")
    decision = await run_claim_verification(request)
    print(decision.model_dump_json(indent=2))
```

**Multi-Agent Pattern:** Manual sequential pipeline
**Handoff Style:** Pass Pydantic model from Agent 1 as input to Agent 2
**Key Advantage:** Full type safety with Pydantic models for inputs/outputs

---

### 4. OpenAI Agents SDK (formerly Swarm)
**Official OpenAI multi-agent framework.**

```bash
pip install openai-agents
```

**Key Concepts:**
- `Agent` - Defines agent with instructions and tools
- `Swarm` - Orchestrates multiple agents
- `handoff()` - Transfer to another agent
- Function tools with docstrings

**Documentation:** https://github.com/openai/openai-agents-python

**Implementation Notes:**
- Lightweight and simple
- OpenAI-specific (GPT models only)
- Good for agent handoffs
- Stateless by design

```python
# Example structure for OpenAI Agents SDK with TWO AGENTS
from openai_agents import Agent, Swarm, handoff

# --- TOOLS FOR WEATHER AGENT ---
def geocode_location(city: str, state: str, postcode: str) -> dict:
    """Convert address to latitude/longitude coordinates using Nominatim."""
    import requests
    query = f"{city}, {state}, {postcode}, Australia"
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "countrycodes": "au"},
        headers={"User-Agent": "WeatherVerificationAgent/1.0"}
    )
    data = response.json()
    if data:
        return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
    return {"error": "Location not found"}

def fetch_bom_weather(lat: float, lon: float, date: str, state: str) -> dict:
    """Fetch weather observations from Australian Bureau of Meteorology."""
    import requests
    response = requests.get(
        "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
        params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
    )
    # Parse HTML response... (simplified)
    return {"thunderstorms": "Observed", "strong_wind": "Observed"}

# --- HANDOFF FUNCTION ---
def transfer_to_eligibility_agent():
    """Transfer to the Claims Eligibility Agent for final decision."""
    return eligibility_agent

# --- AGENT 1: Weather Verification ---
weather_agent = Agent(
    name="Weather Verification Agent",
    instructions="""You are a Weather Verification Agent. Your job is to:
1. Use geocode_location to convert the address to coordinates
2. Use fetch_bom_weather to get weather observations
3. Summarize what weather was observed (thunderstorms, strong wind)
4. Then ALWAYS call transfer_to_eligibility_agent to hand off for the final decision

Do not make eligibility decisions yourself - always hand off after gathering weather data.""",
    tools=[geocode_location, fetch_bom_weather, transfer_to_eligibility_agent]
)

# --- AGENT 2: Claims Eligibility (no tools except optional handoff) ---
eligibility_agent = Agent(
    name="Claims Eligibility Agent",
    instructions="""You are a Claims Eligibility Agent. You receive weather verification 
results from the Weather Verification Agent and make the final CAT eligibility decision.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ Decision: APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT â†’ Decision: REVIEW
- Neither "Observed" = NOT A CAT EVENT â†’ Decision: DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
- Date must not be in the future

Provide your decision in this format:
- CAT Status: [CONFIRMED/POSSIBLE/NOT_CAT]
- Decision: [APPROVED/REVIEW/DENIED]
- Confidence: [HIGH/MEDIUM/LOW]
- Reasoning: [Your explanation]
- Next Steps: [What should happen next]""",
    tools=[]  # No tools - pure LLM reasoning
)

# --- ORCHESTRATION: Swarm with handoffs ---
client = Swarm()

def run_claim_verification(city: str, state: str, postcode: str, date: str):
    """Run the multi-agent verification flow."""
    messages = [
        {"role": "user", "content": f"Verify weather and determine CAT eligibility for: {city}, {state}, {postcode} on date {date}"}
    ]
    
    # Start with weather agent, will auto-handoff to eligibility agent
    response = client.run(
        agent=weather_agent,
        messages=messages
    )
    
    return response

# Usage
result = run_claim_verification("Brisbane", "QLD", "4000", "2025-03-07")
print(result.messages[-1]["content"])  # Final eligibility decision
```

**Multi-Agent Pattern:** Explicit handoff via `transfer_to_*` functions
**Handoff Style:** Agent returns another agent to transfer control
**Key Advantage:** Very simple and intuitive handoff pattern

---

### 5. Anthropic Claude Tool Use
**Native Claude tool calling (not a full framework, but important to evaluate).**

```bash
pip install anthropic
```

**Key Concepts:**
- `tools` parameter in API call
- Tool definitions with JSON schema
- `tool_use` and `tool_result` message types
- Manual orchestration required

**Documentation:** https://docs.anthropic.com/en/docs/build-with-claude/tool-use

**Implementation Notes:**
- Not a framework - raw API
- Maximum flexibility
- Requires building your own orchestration
- Best Claude model support

```python
# Example structure for Anthropic Tool Use with TWO AGENTS
import anthropic
import json

client = anthropic.Anthropic()

# --- TOOL DEFINITIONS ---
weather_tools = [
    {
        "name": "geocode_location",
        "description": "Convert address to latitude/longitude coordinates using Nominatim",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "state": {"type": "string", "description": "Australian state code (e.g., QLD, NSW)"},
                "postcode": {"type": "string", "description": "Postcode"}
            },
            "required": ["city", "state", "postcode"]
        }
    },
    {
        "name": "fetch_bom_weather",
        "description": "Fetch weather observations from Australian Bureau of Meteorology",
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

# --- TOOL EXECUTION ---
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "geocode_location":
        # Implementation using shared_utils
        return json.dumps({"latitude": -27.47, "longitude": 153.03})
    elif tool_name == "fetch_bom_weather":
        # Implementation using shared_utils
        return json.dumps({"thunderstorms": "Observed", "strong_wind": "Observed"})
    return json.dumps({"error": f"Unknown tool: {tool_name}"})

# --- AGENT 1: Weather Verification ---
def run_weather_agent(city: str, state: str, postcode: str, date: str) -> dict:
    """Run weather verification agent with tool loop."""
    messages = [
        {"role": "user", "content": f"Verify weather conditions for {city}, {state}, {postcode} on {date}. First geocode the location, then fetch BOM weather data."}
    ]
    
    system_prompt = """You are a Weather Verification Agent. Use the provided tools to:
1. Geocode the location to get coordinates
2. Fetch weather data from BOM for that location and date
3. Report your findings in a structured JSON format with: location, coordinates, date, weather_events, severe_weather_confirmed, reasoning"""

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            tools=weather_tools,
            messages=messages
        )
        
        # Check if we need to handle tool use
        if response.stop_reason == "tool_use":
            # Extract tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Agent is done - extract final response
            final_text = next((b.text for b in response.content if hasattr(b, 'text')), "")
            return {"weather_result": final_text, "messages": messages}

# --- AGENT 2: Claims Eligibility (no tools) ---
def run_eligibility_agent(weather_result: str) -> dict:
    """Run eligibility agent - pure LLM reasoning, no tools."""
    
    system_prompt = """You are a Claims Eligibility Agent. You receive weather verification results and determine CAT event eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT â†’ REVIEW
- Neither "Observed" = NOT A CAT EVENT â†’ DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
- Date must not be in the future

Respond with JSON: {"cat_event_status": "...", "eligibility_decision": "...", "confidence": "...", "reasoning": "...", "next_steps": [...]}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Evaluate CAT eligibility based on this weather verification:\n\n{weather_result}"}
        ]
    )
    
    return {"eligibility_result": response.content[0].text}

# --- ORCHESTRATION: Manual Pipeline ---
def run_claim_verification(city: str, state: str, postcode: str, date: str) -> dict:
    """Orchestrate both agents in sequence."""
    
    # Step 1: Weather verification (with tools)
    weather_output = run_weather_agent(city, state, postcode, date)
    
    # Step 2: Eligibility decision (LLM reasoning)
    eligibility_output = run_eligibility_agent(weather_output["weather_result"])
    
    return {
        "weather_verification": weather_output["weather_result"],
        "eligibility_decision": eligibility_output["eligibility_result"]
    }

# Usage
result = run_claim_verification("Brisbane", "QLD", "4000", "2025-03-07")
print(result["eligibility_decision"])
```

**Multi-Agent Pattern:** Manual sequential orchestration
**Handoff Style:** Pass output string from Agent 1 as input to Agent 2
**Key Advantage:** Maximum flexibility, best Claude model support, no framework lock-in

---

### 6. Haystack
**By deepset. Strong for RAG pipelines, now supports agents.**

```bash
pip install haystack-ai
```

**Key Concepts:**
- `Pipeline` - Connect components
- `Component` - Building blocks (retrievers, generators, etc.)
- `Agent` - LLM agent with tools
- `Tool` - Callable tools

**Documentation:** https://docs.haystack.deepset.ai/

**Implementation Notes:**
- Originally for search/RAG
- Agent support added recently
- Good for document-heavy workflows
- Strong enterprise features

```python
# Example structure for Haystack with TWO AGENTS
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.agents import Agent, Tool
from typing import Optional
import requests

# --- TOOLS FOR WEATHER AGENT ---
def geocode_location(city: str, state: str, postcode: str) -> dict:
    """Convert address to latitude/longitude coordinates using Nominatim."""
    query = f"{city}, {state}, {postcode}, Australia"
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "countrycodes": "au"},
        headers={"User-Agent": "WeatherVerificationAgent/1.0"}
    )
    data = response.json()
    if data:
        return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
    return {"error": "Location not found"}

def fetch_bom_weather(lat: float, lon: float, date: str, state: str) -> dict:
    """Fetch weather observations from Australian Bureau of Meteorology."""
    response = requests.get(
        "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
        params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
    )
    # Parse HTML... (simplified)
    return {"thunderstorms": "Observed", "strong_wind": "Observed"}

weather_tools = [
    Tool(name="geocode_location", function=geocode_location, 
         description="Convert address to lat/lon coordinates"),
    Tool(name="fetch_bom_weather", function=fetch_bom_weather,
         description="Fetch weather data from BOM")
]

# --- AGENT 1: Weather Verification ---
weather_agent = Agent(
    generator=OpenAIChatGenerator(model="gpt-4"),
    tools=weather_tools,
    system_prompt="""You are a Weather Verification Agent. Use your tools to:
1. Geocode the location to coordinates
2. Fetch weather observations from BOM
3. Report findings as JSON: {location, coordinates, weather_events, severe_weather_confirmed, reasoning}"""
)

# --- AGENT 2: Claims Eligibility (no tools) ---
eligibility_generator = OpenAIChatGenerator(model="gpt-4")

eligibility_prompt_builder = ChatPromptBuilder(
    template=[
        ChatMessage.from_system("""You are a Claims Eligibility Agent. Based on weather verification results, determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ APPROVED
- Only ONE = POSSIBLE CAT â†’ REVIEW
- Neither = NOT CAT â†’ DENIED

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning, next_steps}"""),
        ChatMessage.from_user("Weather verification result:\n{{weather_result}}\n\nMake your eligibility decision.")
    ]
)

# --- PIPELINE: Connect both agents ---
# Option 1: Sequential pipeline with custom component

from haystack import component

@component
class WeatherAgentComponent:
    def __init__(self, agent: Agent):
        self.agent = agent
    
    @component.output_types(weather_result=str)
    def run(self, query: str):
        result = self.agent.run(messages=[ChatMessage.from_user(query)])
        return {"weather_result": result["replies"][-1].text}

@component  
class EligibilityAgentComponent:
    def __init__(self, generator, prompt_builder):
        self.generator = generator
        self.prompt_builder = prompt_builder
    
    @component.output_types(eligibility_decision=str)
    def run(self, weather_result: str):
        prompt = self.prompt_builder.run(weather_result=weather_result)
        result = self.generator.run(messages=prompt["prompt"])
        return {"eligibility_decision": result["replies"][-1].text}

# Build the multi-agent pipeline
pipeline = Pipeline()
pipeline.add_component("weather_agent", WeatherAgentComponent(weather_agent))
pipeline.add_component("eligibility_agent", EligibilityAgentComponent(eligibility_generator, eligibility_prompt_builder))

# Connect agents
pipeline.connect("weather_agent.weather_result", "eligibility_agent.weather_result")

# Run pipeline
def run_claim_verification(city: str, state: str, postcode: str, date: str):
    result = pipeline.run({
        "weather_agent": {"query": f"Verify weather for {city}, {state}, {postcode} on {date}"}
    })
    return result["eligibility_agent"]["eligibility_decision"]

# Usage
decision = run_claim_verification("Brisbane", "QLD", "4000", "2025-03-07")
print(decision)
```

**Multi-Agent Pattern:** Pipeline with custom components wrapping agents
**Handoff Style:** Pipeline connections pass data between components
**Key Advantage:** Strong for workflows mixing RAG + agents

---

### 7. Azure Semantic Kernel
**Microsoft's SDK for AI orchestration. Enterprise-focused.**

```bash
pip install semantic-kernel
```

**Key Concepts:**
- `Kernel` - Central orchestrator
- `Plugin` - Collection of functions
- `Function` - Individual capabilities
- `Planner` - Auto-orchestration
- `Memory` - Context management

**Documentation:** https://learn.microsoft.com/en-us/semantic-kernel/

**Implementation Notes:**
- Strong Azure integration
- Supports multiple LLM providers
- Good for enterprise scenarios
- Can be verbose

```python
# Example structure for Semantic Kernel with TWO AGENTS
import semantic_kernel as sk
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
import requests
import json

# --- WEATHER VERIFICATION PLUGIN (Tools for Agent 1) ---
class WeatherVerificationPlugin:
    """Plugin containing tools for weather verification."""
    
    @kernel_function(description="Convert address to latitude/longitude coordinates using Nominatim")
    def geocode_location(self, city: str, state: str, postcode: str) -> str:
        query = f"{city}, {state}, {postcode}, Australia"
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "countrycodes": "au"},
            headers={"User-Agent": "WeatherVerificationAgent/1.0"}
        )
        data = response.json()
        if data:
            return json.dumps({"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])})
        return json.dumps({"error": "Location not found"})
    
    @kernel_function(description="Fetch weather observations from Australian Bureau of Meteorology")
    def fetch_bom_weather(self, lat: float, lon: float, date: str, state: str) -> str:
        response = requests.get(
            "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
            params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
        )
        # Parse HTML... (simplified)
        return json.dumps({"thunderstorms": "Observed", "strong_wind": "Observed"})

# --- SETUP KERNEL ---
kernel = sk.Kernel()

# Add AI service (OpenAI or Azure OpenAI)
kernel.add_service(OpenAIChatCompletion(service_id="gpt4", ai_model_id="gpt-4"))
# Or for Azure: kernel.add_service(AzureChatCompletion(service_id="gpt4", deployment_name="gpt-4", endpoint=..., api_key=...))

# Add weather plugin
kernel.add_plugin(WeatherVerificationPlugin(), plugin_name="weather")

# --- AGENT 1: Weather Verification (with function calling) ---
async def run_weather_agent(city: str, state: str, postcode: str, date: str) -> str:
    """Run weather verification agent with tool access."""
    
    chat_history = ChatHistory()
    chat_history.add_system_message("""You are a Weather Verification Agent. Use your available functions to:
1. Geocode the location to get coordinates
2. Fetch weather observations from BOM
3. Report findings as JSON: {location, coordinates, weather_events, severe_weather_confirmed, reasoning}

Always use the tools - do not make up data.""")
    
    chat_history.add_user_message(f"Verify weather for {city}, {state}, {postcode} on {date}")
    
    # Enable auto function calling
    settings = kernel.get_prompt_execution_settings_from_service_id("gpt4")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    result = await kernel.invoke_prompt(
        prompt="{{$chat_history}}",
        chat_history=chat_history,
        settings=settings
    )
    
    return str(result)

# --- AGENT 2: Claims Eligibility (no tools, semantic function) ---
eligibility_prompt = """You are a Claims Eligibility Agent. Based on the weather verification results below, 
determine CAT event eligibility.

WEATHER VERIFICATION:
{{$weather_result}}

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ APPROVED
- Only ONE = POSSIBLE CAT â†’ REVIEW  
- Neither = NOT CAT â†’ DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
- Date must not be in the future

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning, next_steps}"""

async def run_eligibility_agent(weather_result: str) -> str:
    """Run eligibility agent - pure LLM reasoning."""
    
    result = await kernel.invoke_prompt(
        prompt=eligibility_prompt,
        weather_result=weather_result
    )
    
    return str(result)

# --- ORCHESTRATION: Sequential execution ---
async def run_claim_verification(city: str, state: str, postcode: str, date: str) -> dict:
    """Orchestrate both agents."""
    
    # Step 1: Weather verification (with tools)
    weather_result = await run_weather_agent(city, state, postcode, date)
    
    # Step 2: Eligibility decision (semantic function)
    eligibility_result = await run_eligibility_agent(weather_result)
    
    return {
        "weather_verification": weather_result,
        "eligibility_decision": eligibility_result
    }

# Usage
import asyncio
result = asyncio.run(run_claim_verification("Brisbane", "QLD", "4000", "2025-03-07"))
print(result["eligibility_decision"])
```

**Multi-Agent Pattern:** Kernel with plugins + semantic functions
**Handoff Style:** Pass string output between function calls
**Key Advantage:** Strong Azure integration, enterprise features, Planner can auto-orchestrate

---

### 8. CrewAI
**Popular for role-based multi-agent systems.**

```bash
pip install crewai crewai-tools
```

**Key Concepts:**
- `Agent` - Role-based agent definition
- `Task` - Specific objectives
- `Crew` - Team of agents
- `Process` - Sequential or hierarchical
- `Tool` - Agent capabilities

**Documentation:** https://docs.crewai.com/

**Implementation Notes:**
- Very intuitive API
- Good for collaborative agents
- Less flexible orchestration than LangGraph
- Active development

```python
# Example structure for CrewAI with TWO AGENTS
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import requests

# --- TOOLS (only for Weather Agent) ---
@tool("Geocode Location")
def geocode_location(city: str, state: str, postcode: str) -> dict:
    """Convert address to latitude/longitude coordinates using Nominatim.
    
    Args:
        city: City name (e.g., "Brisbane")
        state: Australian state code (e.g., "QLD")
        postcode: Postcode (e.g., "4000")
    """
    query = f"{city}, {state}, {postcode}, Australia"
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "countrycodes": "au"},
        headers={"User-Agent": "WeatherVerificationAgent/1.0"}
    )
    data = response.json()
    if data:
        return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
    return {"error": "Location not found"}

@tool("Fetch BOM Weather")
def fetch_bom_weather(lat: float, lon: float, date: str, state: str) -> dict:
    """Fetch weather observations from Australian Bureau of Meteorology.
    
    Args:
        lat: Latitude (e.g., -27.5)
        lon: Longitude (e.g., 153.0)
        date: Date in YYYY-MM-DD format
        state: Australian state code (e.g., "QLD")
    """
    response = requests.get(
        "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
        params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
    )
    # Parse HTML response... (simplified)
    return {"thunderstorms": "Observed", "strong_wind": "Observed"}

# --- AGENT 1: Weather Verification Specialist ---
weather_agent = Agent(
    role="Weather Verification Specialist",
    goal="Accurately verify severe weather events by gathering data from official sources",
    backstory="""You are an expert meteorologist who specializes in verifying weather 
events for insurance claims. You have access to the Australian Bureau of Meteorology 
data and always use your tools to gather factual evidence. You never make up data.""",
    tools=[geocode_location, fetch_bom_weather],
    llm="gpt-4",
    verbose=True
)

# --- AGENT 2: Claims Eligibility Analyst (no tools) ---
eligibility_agent = Agent(
    role="Claims Eligibility Analyst",
    goal="Make accurate CAT event eligibility decisions based on weather evidence",
    backstory="""You are a senior claims analyst specializing in catastrophic event 
eligibility. You carefully evaluate weather verification reports and apply strict 
business rules to determine if claims qualify as CAT events. You are thorough, 
fair, and always explain your reasoning clearly.""",
    tools=[],  # No tools - pure reasoning
    llm="gpt-4",
    verbose=True
)

# --- TASK 1: Weather Verification ---
weather_task = Task(
    description="""Verify weather conditions for the following claim:
    - Location: {city}, {state}, {postcode}
    - Date: {date}
    
    Steps:
    1. Use the Geocode Location tool to convert the address to coordinates
    2. Use the Fetch BOM Weather tool to get official weather observations
    3. Report your findings including: location, coordinates, weather events observed, 
       and whether severe weather (thunderstorms or strong wind) was confirmed""",
    expected_output="""A structured report containing:
    - Location and coordinates
    - Weather observations (thunderstorms: Observed/Not observed, strong wind: Observed/Not observed)
    - Confirmation of whether severe weather occurred
    - Summary reasoning""",
    agent=weather_agent
)

# --- TASK 2: Eligibility Decision ---
eligibility_task = Task(
    description="""Based on the weather verification report from the Weather Verification Specialist,
    determine CAT event eligibility for this claim.
    
    Apply these rules:
    - BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ Decision: APPROVED
    - Only ONE weather type "Observed" = POSSIBLE CAT â†’ Decision: REVIEW
    - Neither "Observed" = NOT A CAT EVENT â†’ Decision: DENIED
    
    Also validate:
    - Coordinates are within Australia (-44 to -10 lat, 112 to 154 lon)
    - Date is not in the future""",
    expected_output="""A JSON eligibility decision containing:
    - cat_event_status: CONFIRMED/POSSIBLE/NOT_CAT
    - eligibility_decision: APPROVED/REVIEW/DENIED
    - confidence: HIGH/MEDIUM/LOW
    - reasoning: Detailed explanation
    - next_steps: List of recommended actions""",
    agent=eligibility_agent,
    context=[weather_task]  # This task depends on weather_task output
)

# --- CREW: Orchestrate the agents ---
claims_crew = Crew(
    agents=[weather_agent, eligibility_agent],
    tasks=[weather_task, eligibility_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True
)

# --- RUN THE CREW ---
def run_claim_verification(city: str, state: str, postcode: str, date: str) -> str:
    """Run the multi-agent verification process."""
    result = claims_crew.kickoff(inputs={
        "city": city,
        "state": state,
        "postcode": postcode,
        "date": date
    })
    return result

# Usage
result = run_claim_verification("Brisbane", "QLD", "4000", "2025-03-07")
print(result)
```

**Multi-Agent Pattern:** Crew with role-based agents and dependent tasks
**Handoff Style:** Task `context` parameter passes output from one task to another
**Key Advantage:** Very intuitive role/goal/backstory paradigm, easy to understand

---

### 9. Smolagents (Hugging Face)
**Lightweight agents from Hugging Face.**

```bash
pip install smolagents
```

**Key Concepts:**
- `CodeAgent` - Writes and executes code
- `ToolCallingAgent` - Uses defined tools
- `Tool` - Simple tool definitions
- `ManagedAgent` - Wrap agents for multi-agent
- Minimal abstraction

**Documentation:** https://huggingface.co/docs/smolagents/

**Implementation Notes:**
- Very lightweight
- Good for local/open-source models
- Less feature-rich
- Simple to understand

```python
# Example structure for Smolagents with TWO AGENTS
from smolagents import ToolCallingAgent, Tool, LiteLLMModel, ManagedAgent
import requests

# --- TOOL DEFINITIONS ---
class GeocodeTool(Tool):
    name = "geocode_location"
    description = "Convert address to latitude/longitude coordinates using Nominatim"
    inputs = {
        "city": {"type": "string", "description": "City name (e.g., Brisbane)"},
        "state": {"type": "string", "description": "Australian state code (e.g., QLD)"},
        "postcode": {"type": "string", "description": "Postcode (e.g., 4000)"}
    }
    output_type = "object"
    
    def forward(self, city: str, state: str, postcode: str) -> dict:
        query = f"{city}, {state}, {postcode}, Australia"
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "countrycodes": "au"},
            headers={"User-Agent": "WeatherVerificationAgent/1.0"}
        )
        data = response.json()
        if data:
            return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
        return {"error": "Location not found"}

class WeatherTool(Tool):
    name = "fetch_bom_weather"
    description = "Fetch weather observations from Australian Bureau of Meteorology"
    inputs = {
        "lat": {"type": "number", "description": "Latitude"},
        "lon": {"type": "number", "description": "Longitude"},
        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        "state": {"type": "string", "description": "Australian state code"}
    }
    output_type = "object"
    
    def forward(self, lat: float, lon: float, date: str, state: str) -> dict:
        response = requests.get(
            "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
            params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
        )
        # Parse HTML... (simplified)
        return {"thunderstorms": "Observed", "strong_wind": "Observed"}

# --- SHARED MODEL ---
model = LiteLLMModel(model_id="gpt-4")

# --- AGENT 1: Weather Verification (with tools) ---
weather_agent = ToolCallingAgent(
    tools=[GeocodeTool(), WeatherTool()],
    model=model,
    system_prompt="""You are a Weather Verification Agent. Use your tools to:
1. Geocode the location to coordinates
2. Fetch weather observations from BOM
3. Report findings as JSON with: location, coordinates, weather_events, severe_weather_confirmed"""
)

# --- AGENT 2: Claims Eligibility (no tools, uses Agent 1 as managed agent) ---
# In smolagents, you can create a ManagedAgent to let one agent call another

managed_weather_agent = ManagedAgent(
    agent=weather_agent,
    name="weather_verification_agent",
    description="Use this agent to verify weather conditions for a location and date. Provide city, state, postcode, and date."
)

eligibility_agent = ToolCallingAgent(
    tools=[managed_weather_agent],  # The weather agent is a "tool" for this agent
    model=model,
    system_prompt="""You are a Claims Eligibility Agent. You orchestrate weather verification 
and make CAT eligibility decisions.

WORKFLOW:
1. Use the weather_verification_agent to get weather data for the claim location
2. Evaluate the results using these rules:
   - BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ APPROVED
   - Only ONE = POSSIBLE CAT â†’ REVIEW
   - Neither = NOT CAT â†’ DENIED

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning, next_steps}"""
)

# --- ALTERNATIVE: Sequential without ManagedAgent ---
def run_claim_verification_sequential(city: str, state: str, postcode: str, date: str) -> dict:
    """Run both agents sequentially."""
    
    # Agent 1: Weather verification
    weather_result = weather_agent.run(
        f"Verify weather for {city}, {state}, {postcode} on {date}"
    )
    
    # Agent 2: Eligibility (no tools, just reasoning)
    # Create a simple agent without tools for pure reasoning
    reasoning_agent = ToolCallingAgent(
        tools=[],
        model=model,
        system_prompt="""You are a Claims Eligibility Agent. Evaluate weather verification results and determine CAT eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ APPROVED
- Only ONE = POSSIBLE CAT â†’ REVIEW
- Neither = NOT CAT â†’ DENIED

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning, next_steps}"""
    )
    
    eligibility_result = reasoning_agent.run(
        f"Evaluate CAT eligibility based on this weather verification:\n\n{weather_result}"
    )
    
    return {
        "weather_verification": weather_result,
        "eligibility_decision": eligibility_result
    }

# --- USAGE: With ManagedAgent (hierarchical) ---
result = eligibility_agent.run(
    "Verify weather and determine CAT eligibility for Brisbane, QLD, 4000 on 2025-03-07"
)
print(result)

# --- USAGE: Sequential ---
result = run_claim_verification_sequential("Brisbane", "QLD", "4000", "2025-03-07")
print(result["eligibility_decision"])
```

**Multi-Agent Pattern:** ManagedAgent (hierarchical) or manual sequential
**Handoff Style:** ManagedAgent lets one agent call another as a tool
**Key Advantage:** Very lightweight, works great with open-source models

---

### 10. LlamaIndex Agents
**Strong for RAG, expanding into agents.**

```bash
pip install llama-index llama-index-agent-openai llama-index-llms-openai
```

**Key Concepts:**
- `FunctionTool` - Wrap functions as tools
- `OpenAIAgent` / `ReActAgent` - Agent types
- `QueryEngine` - For RAG integration
- `AgentRunner` - Execute agents
- Strong document handling

**Documentation:** https://docs.llamaindex.ai/

**Implementation Notes:**
- Best for document-heavy use cases
- Good RAG integration
- Agent features still maturing
- Strong community

```python
# Example structure for LlamaIndex with TWO AGENTS
from llama_index.core.agent import ReActAgent, AgentRunner
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import requests
import json

# --- TOOL FUNCTIONS ---
def geocode_location(city: str, state: str, postcode: str) -> str:
    """Convert address to latitude/longitude coordinates using Nominatim.
    
    Args:
        city: City name (e.g., "Brisbane")
        state: Australian state code (e.g., "QLD")
        postcode: Postcode (e.g., "4000")
    """
    query = f"{city}, {state}, {postcode}, Australia"
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "countrycodes": "au"},
        headers={"User-Agent": "WeatherVerificationAgent/1.0"}
    )
    data = response.json()
    if data:
        return json.dumps({"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])})
    return json.dumps({"error": "Location not found"})

def fetch_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
    """Fetch weather observations from Australian Bureau of Meteorology.
    
    Args:
        lat: Latitude (e.g., -27.5)
        lon: Longitude (e.g., 153.0)
        date: Date in YYYY-MM-DD format
        state: Australian state code (e.g., "QLD")
    """
    response = requests.get(
        "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
        params={"lat": round(lat, 1), "lon": round(lon, 1), "date": date, "state": state, "unique_id": "test"}
    )
    # Parse HTML... (simplified)
    return json.dumps({"thunderstorms": "Observed", "strong_wind": "Observed"})

# --- TOOLS FOR WEATHER AGENT ---
weather_tools = [
    FunctionTool.from_defaults(
        fn=geocode_location,
        name="geocode_location",
        description="Convert address to latitude/longitude coordinates"
    ),
    FunctionTool.from_defaults(
        fn=fetch_bom_weather,
        name="fetch_bom_weather",
        description="Fetch weather observations from Australian Bureau of Meteorology"
    ),
]

# --- LLM ---
llm = OpenAI(model="gpt-4", temperature=0)

# --- AGENT 1: Weather Verification (ReAct agent with tools) ---
weather_agent = ReActAgent.from_tools(
    tools=weather_tools,
    llm=llm,
    verbose=True,
    system_prompt="""You are a Weather Verification Agent. Your job is to:
1. Use geocode_location to convert the address to coordinates
2. Use fetch_bom_weather to get official weather observations
3. Report your findings as JSON: {location, coordinates, date, weather_events: {thunderstorms, strong_wind}, severe_weather_confirmed, reasoning}

Always use your tools to get real data. Never make up coordinates or weather observations."""
)

# --- AGENT 2: Claims Eligibility (no tools, reasoning only) ---
# For a no-tools reasoning agent, we can use a simple LLM call or a ReActAgent with empty tools

ELIGIBILITY_SYSTEM_PROMPT = """You are a Claims Eligibility Agent. You receive weather verification results 
and determine CAT event eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT â†’ APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT â†’ REVIEW
- Neither "Observed" = NOT A CAT EVENT â†’ DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
- Date must not be in the future

Respond with JSON: {cat_event_status, eligibility_decision, confidence, reasoning, next_steps}"""

eligibility_agent = ReActAgent.from_tools(
    tools=[],  # No tools - pure reasoning
    llm=llm,
    verbose=True,
    system_prompt=ELIGIBILITY_SYSTEM_PROMPT
)

# --- ORCHESTRATION: Sequential Pipeline ---
def run_claim_verification(city: str, state: str, postcode: str, date: str) -> dict:
    """Run both agents in sequence."""
    
    # Step 1: Weather verification (with tools)
    weather_query = f"Verify weather conditions for {city}, {state}, {postcode} on {date}"
    weather_response = weather_agent.chat(weather_query)
    weather_result = str(weather_response)
    
    # Step 2: Eligibility decision (reasoning only)
    eligibility_query = f"Evaluate CAT eligibility based on this weather verification:\n\n{weather_result}"
    eligibility_response = eligibility_agent.chat(eligibility_query)
    
    return {
        "weather_verification": weather_result,
        "eligibility_decision": str(eligibility_response)
    }

# --- ALTERNATIVE: Using AgentRunner for more control ---
from llama_index.core.agent import AgentRunner

def run_with_agent_runner(city: str, state: str, postcode: str, date: str) -> dict:
    """Use AgentRunner for step-by-step execution control."""
    
    # Weather agent with runner
    weather_runner = AgentRunner(agent_worker=weather_agent)
    
    # Create task and run to completion
    weather_task = weather_runner.create_task(
        f"Verify weather for {city}, {state}, {postcode} on {date}"
    )
    
    # Step through (or run_step until done)
    weather_result = None
    while not weather_task.is_done:
        step_output = weather_runner.run_step(weather_task.task_id)
        if step_output.is_last:
            weather_result = str(step_output.output)
    
    # Eligibility agent
    eligibility_task = f"Evaluate: {weather_result}"
    eligibility_response = eligibility_agent.chat(eligibility_task)
    
    return {
        "weather_verification": weather_result,
        "eligibility_decision": str(eligibility_response)
    }

# --- USAGE ---
result = run_claim_verification("Brisbane", "QLD", "4000", "2025-03-07")
print("Weather:", result["weather_verification"])
print("Eligibility:", result["eligibility_decision"])
```

**Multi-Agent Pattern:** Sequential with separate ReActAgent instances
**Handoff Style:** Pass string output from Agent 1 as input to Agent 2
**Key Advantage:** Excellent for combining agents with document retrieval (RAG)

---

## Shared Utility Code

All implementations should use these shared utilities:

```python
# shared_utils.py
import httpx
from bs4 import BeautifulSoup
from typing import Optional
import asyncio

async def geocode_address(city: str, state: str, postcode: str) -> Optional[dict]:
    """
    Geocode an Australian address using Nominatim.
    Returns dict with latitude and longitude, or None if not found.
    """
    async with httpx.AsyncClient() as client:
        query = f"{city}, {state}, {postcode}, Australia"
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "countrycodes": "au"},
            headers={"User-Agent": "WeatherVerificationAgent/1.0"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return {
                    "latitude": float(data[0]["lat"]),
                    "longitude": float(data[0]["lon"])
                }
    return None


async def fetch_bom_observations(lat: float, lon: float, date: str, state: str) -> dict:
    """
    Fetch storm observations from BOM.
    Date format: YYYY-MM-DD
    Returns dict with thunderstorms and strong_wind status.
    """
    # Round coordinates to 1 decimal place as BOM expects
    lat = round(lat, 1)
    lon = round(lon, 1)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py",
            params={
                "lat": lat,
                "lon": lon,
                "date": date,
                "state": state,
                "location": f"{lat},{lon}",
                "unique_id": "eval_test"
            }
        )
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for invalid location
            if "Invalid location" in response.text:
                return {"error": "Invalid location - must be mainland Australia"}
            
            # Parse the table
            result = {
                "thunderstorms": "No reports or observations",
                "strong_wind": "No reports or observations"
            }
            
            table = soup.find('table', class_='table')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        event_type = cols[0].get_text(strip=True).lower()
                        status = cols[1].get_text(strip=True)
                        
                        if "thunderstorm" in event_type:
                            result["thunderstorms"] = status
                        elif "strong wind" in event_type:
                            result["strong_wind"] = status
            
            return result
        
        return {"error": f"BOM API returned status {response.status_code}"}


def is_severe_weather(weather_data: dict) -> bool:
    """Determine if weather data indicates severe weather."""
    if "error" in weather_data:
        return False
    
    return (
        weather_data.get("thunderstorms") == "Observed" or
        weather_data.get("strong_wind") == "Observed"
    )
```

---

## Implementation Checklist

For each framework, create a file named `{framework}_weather_agent.py` that:

- [ ] Imports necessary dependencies
- [ ] Defines the geocoding tool
- [ ] Defines the BOM weather fetch tool
- [ ] Creates an agent with appropriate system prompt
- [ ] Handles the full flow: geocode â†’ fetch weather â†’ analyze â†’ respond
- [ ] Returns structured output (preferably Pydantic model or typed dict)
- [ ] Includes error handling
- [ ] Can be run standalone with example input

### Test Cases to Run

```python
test_cases = [
    # Case 1: Known severe weather event (Cyclone Alfred area)
    {"city": "Brisbane", "state": "QLD", "postcode": "4000", "date": "2025-03-07"},
    
    # Case 2: Regional area
    {"city": "Mcdowall", "state": "QLD", "postcode": "4053", "date": "2025-03-07"},
    
    # Case 3: Different state
    {"city": "Sydney", "state": "NSW", "postcode": "2000", "date": "2025-03-07"},
    
    # Case 4: Likely no severe weather
    {"city": "Perth", "state": "WA", "postcode": "6000", "date": "2025-01-15"},
]
```

---

## Output Format

Each implementation should produce a summary report:

```markdown
## {Framework Name} Evaluation

### Setup
- Dependencies installed: {list}
- Lines of code: {count}
- Setup complexity: {Easy/Medium/Hard}

### Implementation Notes
- {Key observations}
- {Challenges encountered}
- {Workarounds needed}

### Results
| Test Case | Result | Execution Time |
|-----------|--------|----------------|
| Brisbane 2025-03-07 | Severe weather: Yes | 2.3s |
| ... | ... | ... |

### Evaluation Scores (1-5)
| Criterion | Score | Notes |
|-----------|-------|-------|
| Setup Complexity | X | ... |
| Tool Definition | X | ... |
| Type Safety | X | ... |
| Error Handling | X | ... |
| Documentation | X | ... |
| Multi-Agent Support | X | ... |
| Production Readiness | X | ... |

### Recommendation
{Would/Would not recommend for the Nemo expansion because...}
```

---

## Framework Comparison Matrix (To Fill In)

| Framework | Setup | Tools | Types | Orchestration | Multi-Agent | Streaming | Models | Maturity |
|-----------|-------|-------|-------|---------------|-------------|-----------|--------|----------|
| AutoGen 0.4 | | | | | | | | |
| LangGraph | | | | | | | | |
| Pydantic AI | | | | | | | | |
| OpenAI Agents | | | | | | | | |
| Anthropic | | | | | | | | |
| Haystack | | | | | | | | |
| Semantic Kernel | | | | | | | | |
| CrewAI | | | | | | | | |
| Smolagents | | | | | | | | |
| LlamaIndex | | | | | | | | |

---

## Multi-Agent Pattern Comparison

| Framework | Pattern | Handoff Mechanism | Conditional Routing | Shared State |
|-----------|---------|-------------------|---------------------|--------------|
| **AutoGen 0.4** | Group Chat | Implicit (shared conversation) | SelectorGroupChat | Chat history |
| **LangGraph** | State Machine | Explicit (typed state) | `add_conditional_edges` | TypedDict state |
| **Pydantic AI** | Manual Pipeline | Pydantic models | Custom code | Via function args |
| **OpenAI Agents** | Transfer Functions | `transfer_to_agent()` | Via transfer logic | Context variables |
| **Anthropic** | Manual Loop | String passing | Custom code | Message list |
| **Haystack** | Pipeline | Component connections | Custom components | Pipeline data |
| **Semantic Kernel** | Kernel Functions | String/object passing | Planner or custom | Kernel context |
| **CrewAI** | Task Dependencies | `context` parameter | Process type | Shared memory |
| **Smolagents** | ManagedAgent | Agent-as-tool | Custom code | Agent output |
| **LlamaIndex** | Sequential Agents | String passing | Custom code | Chat history |

### Key Insights:

**Best for Complex Conditional Flows:**
- **LangGraph** - Explicit state machine with `add_conditional_edges`
- **Semantic Kernel** - Planner can auto-orchestrate based on goals

**Best for Simple Handoffs:**
- **OpenAI Agents** - Intuitive `transfer_to_*` pattern
- **CrewAI** - Task `context` parameter is very clean

**Best for Type Safety:**
- **Pydantic AI** - Full Pydantic model validation between agents
- **LangGraph** - TypedDict state gives some typing

**Best for Enterprise/Azure:**
- **Semantic Kernel** - Deep Azure integration
- **AutoGen** - Microsoft ecosystem

**Most Lightweight:**
- **Smolagents** - Minimal dependencies
- **Anthropic** - Raw API, zero framework overhead

---

## Evaluation Phases Overview

The evaluation is structured in three phases of increasing complexity. This ensures we don't over-invest in frameworks that fail basic tests, while also properly testing the advanced capabilities needed for production.

### Why Multi-Phase Evaluation?

A framework that handles 2 sequential agents elegantly might completely fall apart when you need complex routing:

```
                    ┌─→ Storm Agent ────→ Building Assessment ─┐
                    │                            ↓              │
Intake → Router ────┼─→ Food Spoilage ──→ CAT Check ──────────┼─→ Fraud → Amount → Audit
                    │        ↑                                  │
                    │        └── [retry if BOM fails] ─────────┘
                    └─→ Theft Agent ─────→ Police Report ──────┘
```

### Capability Coverage Matrix

| Capability | Phase 1 (Basic) | Phase 2 (Complex) | Phase 3 (Prototype) | Why It Matters |
|------------|-----------------|-------------------|---------------------|----------------|
| Basic tool definition | ✅ | ✅ | ✅ | Foundation |
| Simple sequential handoff | ✅ | ✅ | ✅ | Foundation |
| Data passing between agents | ✅ | ✅ | ✅ | Foundation |
| Tool-using vs reasoning-only | ✅ | ✅ | ✅ | Different agent types |
| Basic developer experience | ✅ | ✅ | ✅ | Team adoption |
| Conditional routing | ❌ | ✅ | ✅ | "If fraud score > 7, skip to rejection" |
| Parallel execution | ❌ | ✅ | ✅ | "Run coverage AND weather check simultaneously" |
| Loops / cycles | ❌ | ✅ | ✅ | "Agent B asks Agent A for clarification" |
| Error recovery | ❌ | ✅ | ✅ | "BOM API fails, retry or fallback" |
| 4+ agents | ❌ | ✅ | ✅ | Coordination complexity |
| Human-in-the-loop | ❌ | ❌ | ✅ | "Pause for human approval" |
| Dynamic routing | ❌ | ❌ | ✅ | "Route to Storm Agent OR Theft Agent" |
| Observability at scale | ❌ | ✅ | ✅ | Debugging 10 agents talking |

---

## Phase 1: Basic Example (Quick Filter)

**Time Estimate:** 1-2 days per framework

**Purpose:** Eliminate frameworks that are painful even for simple cases.

**What to Build:** The 2-agent Weather Verification → Claims Eligibility flow documented above.

**Pass/Fail Criteria:**
- [ ] Can define tools with reasonable syntax
- [ ] Can create agents with system prompts
- [ ] Agent 1 can call tools and return structured output
- [ ] Agent 2 receives Agent 1's output and reasons over it
- [ ] Code is readable and maintainable
- [ ] Documentation is sufficient to implement

**Frameworks to Eliminate If:**
- Tool definition syntax is excessively verbose
- Multi-agent setup requires excessive boilerplate
- Documentation is insufficient or outdated
- Framework feels abandoned or unstable

---

## Phase 2: Complex Scenarios (Top 3-4 Frameworks)

**Time Estimate:** 2-3 days per framework

**Purpose:** Test real-world complexity that production systems require.

**Pre-requisite:** Pass Phase 1 evaluation

### Scenario 2.1: Conditional Routing

**Test:** Add a Router Agent that decides whether to call Weather Agent or skip directly to Eligibility.

**Business Logic:**
```
IF claim has CatastropheNumber (e.g., "E75")
  → Skip Weather Agent, go directly to Eligibility with CAT confirmed
ELSE
  → Call Weather Agent to verify CAT event
```

**Architecture:**
```
                         ┌─→ Weather Agent ──┐
User Query → Router ─────┤                   ├──→ Eligibility Agent
                         └─→ [Skip weather] ─┘
```

**Implementation Specification:**

```python
# Router Agent System Prompt
ROUTER_SYSTEM_PROMPT = """You are a Claims Router Agent. Your job is to analyze incoming 
claims and route them to the appropriate processing path.

ROUTING RULES:
1. If the claim data includes a non-empty "CatastropheNumber" field (e.g., "E75"):
   - CAT event is already confirmed by official records
   - Route to: SKIP_WEATHER
   - Pass along: {cat_confirmed: true, cat_number: <value>, reason: "Official CAT number provided"}

2. If the claim data has NO CatastropheNumber or it's empty:
   - CAT event needs verification via weather data
   - Route to: VERIFY_WEATHER
   - Pass along: {city, state, postcode, date} for weather verification

ALWAYS respond with a JSON routing decision:
{
  "route": "SKIP_WEATHER" | "VERIFY_WEATHER",
  "reason": "<explanation>",
  "payload": {<data to pass to next agent>}
}
"""

# Test Cases for Conditional Routing
routing_test_cases = [
    # Case 1: Has CAT number - should skip weather
    {
        "input": {
            "city": "Brisbane", "state": "QLD", "postcode": "4000",
            "date": "2025-03-07", "CatastropheNumber": "E75"
        },
        "expected_route": "SKIP_WEATHER"
    },
    # Case 2: No CAT number - should verify weather
    {
        "input": {
            "city": "Brisbane", "state": "QLD", "postcode": "4000",
            "date": "2025-03-07", "CatastropheNumber": ""
        },
        "expected_route": "VERIFY_WEATHER"
    },
    # Case 3: Missing CAT field entirely - should verify weather
    {
        "input": {
            "city": "Sydney", "state": "NSW", "postcode": "2000",
            "date": "2025-03-07"
        },
        "expected_route": "VERIFY_WEATHER"
    },
]
```

**Evaluation Questions:**
- How natural is the conditional routing syntax?
- Can you easily add more routes (e.g., MANUAL_REVIEW)?
- How is routing state preserved for debugging?
- Can routes be changed at runtime?

---

### Scenario 2.2: Error Handling & Retry Logic

**Test:** Handle BOM API failures gracefully with retry and fallback.

**Business Logic:**
```
TRY fetch BOM weather data
  → On success: Continue to Eligibility
  → On first failure: Retry once after 2 seconds
  → On second failure: Route to MANUAL_REVIEW with error context
```

**Architecture:**
```
Weather Agent ──→ [BOM API Call]
                      │
                      ├─→ Success ──→ Eligibility Agent
                      │
                      └─→ Failure ──→ [Retry Logic]
                                          │
                                          ├─→ Retry Success ──→ Eligibility Agent  
                                          │
                                          └─→ Retry Failure ──→ Manual Review Queue
```

**Implementation Specification:**

```python
# Error handling wrapper for BOM tool
class BOMAPIError(Exception):
    """Raised when BOM API fails after retries."""
    def __init__(self, message: str, attempts: int, last_error: str):
        self.message = message
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)

async def fetch_bom_weather_with_retry(
    lat: float, lon: float, date: str, state: str,
    max_retries: int = 2,
    retry_delay: float = 2.0
) -> dict:
    """Fetch BOM weather with retry logic."""
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            result = await fetch_bom_weather(lat, lon, date, state)
            if "error" not in result:
                return result
            last_error = result.get("error", "Unknown error")
        except Exception as e:
            last_error = str(e)
        
        if attempt < max_retries:
            await asyncio.sleep(retry_delay)
    
    # All retries failed
    raise BOMAPIError(
        message=f"BOM API failed after {max_retries + 1} attempts",
        attempts=max_retries + 1,
        last_error=last_error
    )

# Manual Review fallback output
MANUAL_REVIEW_OUTPUT = {
    "route": "MANUAL_REVIEW",
    "reason": "Weather verification failed after retries",
    "error_context": {
        "attempts": 2,
        "last_error": "<error details>",
        "claim_data": "<preserved claim data>"
    },
    "recommended_action": "Manual weather verification required"
}

# Test Cases for Error Handling
error_handling_test_cases = [
    # Case 1: BOM returns error response
    {
        "scenario": "BOM API returns error HTML",
        "mock_response": "<html>Service Unavailable</html>",
        "expected_behavior": "Retry once, then route to MANUAL_REVIEW"
    },
    # Case 2: Network timeout
    {
        "scenario": "Network timeout on first call",
        "mock_behavior": "Timeout after 10s",
        "expected_behavior": "Retry once, succeed on retry, continue normal flow"
    },
    # Case 3: Invalid location
    {
        "scenario": "BOM returns 'Invalid location'",
        "mock_response": "Invalid location - must be mainland Australia",
        "expected_behavior": "Do NOT retry (invalid input, not transient error), route to MANUAL_REVIEW"
    },
]
```

**Evaluation Questions:**
- Does the framework have built-in retry mechanisms?
- How do you distinguish transient vs permanent errors?
- Can you preserve context when routing to fallback?
- How is error state surfaced for debugging?

---

### Scenario 2.3: Agent Clarification Loop

**Test:** Eligibility Agent can request additional information from Weather Agent.

**Business Logic:**
```
IF Eligibility Agent receives ambiguous weather data (only one event type observed)
  → Request Weather Agent to check adjacent dates (+/- 1 day)
  → Weather Agent returns expanded data
  → Eligibility Agent makes final decision with expanded context
```

**Architecture:**
```
Weather Agent ──────────────→ Eligibility Agent
       ↑                            │
       │                            ↓
       │                      [Needs more data?]
       │                            │
       └──── [Check adjacent] ←─────┘
                  dates
```

**Implementation Specification:**

```python
# Clarification Request Schema
class ClarificationRequest:
    """Request from Eligibility Agent to Weather Agent."""
    request_type: str  # "EXPAND_DATE_RANGE" | "VERIFY_COORDINATES" | "CHECK_NEARBY_STATIONS"
    original_query: dict
    additional_params: dict
    reason: str

# Eligibility Agent Enhanced System Prompt
ELIGIBILITY_WITH_CLARIFICATION_PROMPT = """You are a Claims Eligibility Agent with the ability 
to request additional information when needed.

STANDARD EVALUATION:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Neither "Observed" = NOT CAT → DENIED

AMBIGUOUS CASES (only ONE weather type observed):
Before deciding REVIEW, you MAY request clarification:
- Request Weather Agent to check +/- 1 day from incident date
- Request Weather Agent to verify with nearby weather stations

To request clarification, respond with:
{
  "action": "REQUEST_CLARIFICATION",
  "request": {
    "type": "EXPAND_DATE_RANGE",
    "check_dates": ["<date-1>", "<date>", "<date+1>"],
    "reason": "Only thunderstorms observed on incident date, checking adjacent dates for strong wind"
  }
}

After receiving expanded data, make your final decision.
Maximum clarification requests per claim: 2
"""

# Weather Agent Clarification Handler
WEATHER_CLARIFICATION_HANDLER = """When you receive a clarification request:

1. For EXPAND_DATE_RANGE:
   - Fetch BOM data for each date in the request
   - Return combined results with date labels

2. For CHECK_NEARBY_STATIONS:
   - Adjust coordinates by +/- 0.5 degrees
   - Fetch data for original and adjusted locations
   - Return combined results with location labels

Respond with expanded weather data in the same format as original, but include all dates/locations checked.
"""

# Test Cases for Clarification Loop
clarification_test_cases = [
    # Case 1: Ambiguous data triggers clarification
    {
        "initial_weather": {
            "thunderstorms": "Observed",
            "strong_wind": "No reports or observations"
        },
        "expected_action": "REQUEST_CLARIFICATION for adjacent dates",
        "expanded_weather": {
            "2025-03-06": {"thunderstorms": "No reports", "strong_wind": "Observed"},
            "2025-03-07": {"thunderstorms": "Observed", "strong_wind": "No reports"},
            "2025-03-08": {"thunderstorms": "No reports", "strong_wind": "Observed"}
        },
        "expected_decision": "APPROVED (strong wind observed on adjacent days supports CAT event)"
    },
    # Case 2: Clarification doesn't help
    {
        "initial_weather": {
            "thunderstorms": "Observed",
            "strong_wind": "No reports or observations"
        },
        "expanded_weather": {
            "2025-03-06": {"thunderstorms": "No reports", "strong_wind": "No reports"},
            "2025-03-07": {"thunderstorms": "Observed", "strong_wind": "No reports"},
            "2025-03-08": {"thunderstorms": "No reports", "strong_wind": "No reports"}
        },
        "expected_decision": "REVIEW (only isolated thunderstorm activity)"
    },
    # Case 3: Clear data, no clarification needed
    {
        "initial_weather": {
            "thunderstorms": "Observed",
            "strong_wind": "Observed"
        },
        "expected_action": "No clarification, direct APPROVED"
    },
]
```

**Evaluation Questions:**
- How does the framework handle bidirectional communication?
- Can you limit the number of clarification rounds?
- How is the conversation history preserved across loops?
- Is the loop easy to debug and trace?

---

### Scenario 2.4: Third Agent - Fraud Screening

**Test:** Add a Fraud Screening Agent that can short-circuit the flow.

**Business Logic:**
```
AFTER Weather verification, BEFORE final Eligibility:
  → Fraud Agent scores claim (0-10)
  → IF fraud_score >= 7: Route to REJECTED (skip eligibility)
  → IF fraud_score < 7: Continue to Eligibility with fraud_score attached
```

**Architecture:**
```
Weather Agent → Fraud Screening Agent → [Score Check] → Eligibility Agent
                                             │
                                             └─→ [Score >= 7] → REJECTED
```

**Implementation Specification:**

```python
# Fraud Screening Agent
FRAUD_AGENT_SYSTEM_PROMPT = """You are a Fraud Screening Agent for insurance claims. 
Analyze claim data and weather verification results to detect potential fraud indicators.

FRAUD INDICATORS (assign points):
+3: Claim filed within 14 days of policy inception
+2: Weather data shows no severe events but claim alleges storm damage
+2: Location coordinates don't match stated address
+2: Date of incident is suspiciously close to claim filing
+1: High claim amount relative to policy history
+1: Vague or inconsistent description of damage

SCORING:
- 0-3: LOW RISK - Proceed normally
- 4-6: MEDIUM RISK - Flag for review but proceed
- 7-10: HIGH RISK - Recommend rejection

ALWAYS respond with:
{
  "fraud_score": <0-10>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "indicators_found": [
    {"indicator": "<description>", "points": <1-3>}
  ],
  "recommendation": "PROCEED" | "FLAG_FOR_REVIEW" | "REJECT",
  "reasoning": "<explanation>"
}
"""

# Orchestration Logic
def should_reject_claim(fraud_result: dict) -> bool:
    """Determine if claim should be rejected based on fraud score."""
    return fraud_result.get("fraud_score", 0) >= 7

def route_after_fraud(fraud_result: dict) -> str:
    """Determine next step after fraud screening."""
    if should_reject_claim(fraud_result):
        return "REJECTED"
    elif fraud_result.get("risk_level") == "MEDIUM":
        return "ELIGIBILITY_WITH_FLAG"
    else:
        return "ELIGIBILITY"

# Test Cases for Fraud Agent Integration
fraud_test_cases = [
    # Case 1: Clean claim, low fraud score
    {
        "claim_data": {
            "policy_inception": "2024-01-01",
            "incident_date": "2025-03-07",
            "claim_filed": "2025-03-08",
            "amount": 500
        },
        "weather_data": {"thunderstorms": "Observed", "strong_wind": "Observed"},
        "expected_fraud_score": "0-3",
        "expected_route": "ELIGIBILITY"
    },
    # Case 2: Suspicious timing, medium fraud score
    {
        "claim_data": {
            "policy_inception": "2025-03-01",  # Recent policy
            "incident_date": "2025-03-07",     # 6 days after inception
            "claim_filed": "2025-03-08",
            "amount": 500
        },
        "weather_data": {"thunderstorms": "Observed", "strong_wind": "Observed"},
        "expected_fraud_score": "3-5",
        "expected_route": "ELIGIBILITY_WITH_FLAG"
    },
    # Case 3: High fraud indicators, should reject
    {
        "claim_data": {
            "policy_inception": "2025-03-05",  # Very recent
            "incident_date": "2025-03-07",     # 2 days after inception
            "claim_filed": "2025-03-07",       # Same day as incident
            "amount": 500,
            "description": "Food spoiled"      # Vague
        },
        "weather_data": {"thunderstorms": "No reports", "strong_wind": "No reports"},
        "expected_fraud_score": "7+",
        "expected_route": "REJECTED"
    },
]
```

**Evaluation Questions:**
- How easy is it to add a third agent to the existing flow?
- Can you conditionally skip agents based on previous results?
- How do you pass accumulated context (weather + fraud) to final agent?
- How is the short-circuit (early rejection) handled?

---

### Scenario 2.5: Parallel Execution

**Test:** Run geocoding and preliminary checks in parallel.

**Business Logic:**
```
PARALLEL:
  - Task A: Geocode location
  - Task B: Validate claim data format
  - Task C: Check policy status (mock API)
THEN (all complete):
  → Merge results
  → Continue to Weather Agent with validated data
```

**Architecture:**
```
                 ┌─→ Geocode Task ────────┐
                 │                        │
User Query ──────┼─→ Validation Task ─────┼──→ Merge → Weather Agent
                 │                        │
                 └─→ Policy Check Task ───┘
```

**Implementation Specification:**

```python
# Parallel task definitions
async def geocode_task(city: str, state: str, postcode: str) -> dict:
    """Task A: Geocode location."""
    return await geocode_address(city, state, postcode)

async def validation_task(claim_data: dict) -> dict:
    """Task B: Validate claim data format."""
    errors = []
    
    required_fields = ["city", "state", "postcode", "date"]
    for field in required_fields:
        if field not in claim_data or not claim_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate date format
    if "date" in claim_data:
        try:
            datetime.strptime(claim_data["date"], "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid date format: {claim_data['date']}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

async def policy_check_task(policy_number: str) -> dict:
    """Task C: Check policy status (mock)."""
    # Mock implementation - in production would call policy API
    await asyncio.sleep(0.5)  # Simulate API latency
    return {
        "policy_number": policy_number,
        "status": "ACTIVE",
        "inception_date": "2024-01-01",
        "expiry_date": "2025-01-01"
    }

# Parallel execution wrapper
async def run_preliminary_checks(claim_data: dict) -> dict:
    """Run all preliminary checks in parallel."""
    results = await asyncio.gather(
        geocode_task(claim_data["city"], claim_data["state"], claim_data["postcode"]),
        validation_task(claim_data),
        policy_check_task(claim_data.get("policy_number", "UNKNOWN")),
        return_exceptions=True
    )
    
    geocode_result, validation_result, policy_result = results
    
    # Handle any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            return {"error": f"Task {i} failed: {str(result)}"}
    
    # Merge results
    return {
        "coordinates": geocode_result,
        "validation": validation_result,
        "policy": policy_result,
        "all_checks_passed": validation_result.get("valid", False) and policy_result.get("status") == "ACTIVE"
    }

# Test Cases for Parallel Execution
parallel_test_cases = [
    # Case 1: All parallel tasks succeed
    {
        "claim_data": {
            "city": "Brisbane", "state": "QLD", "postcode": "4000",
            "date": "2025-03-07", "policy_number": "POL123"
        },
        "expected": {
            "coordinates": {"latitude": -27.47, "longitude": 153.03},
            "validation": {"valid": True},
            "policy": {"status": "ACTIVE"},
            "all_checks_passed": True
        }
    },
    # Case 2: Validation fails, geocoding succeeds
    {
        "claim_data": {
            "city": "Brisbane", "state": "QLD", "postcode": "4000",
            "date": "invalid-date"
        },
        "expected": {
            "coordinates": {"latitude": -27.47, "longitude": 153.03},
            "validation": {"valid": False, "errors": ["Invalid date format"]},
            "all_checks_passed": False
        }
    },
    # Case 3: One task throws exception
    {
        "claim_data": {
            "city": "InvalidCity123", "state": "XX", "postcode": "0000"
        },
        "expected_behavior": "Geocoding fails, error captured, other tasks complete"
    },
]
```

**Evaluation Questions:**
- Does the framework have native parallel execution support?
- How are parallel task results merged?
- How do you handle partial failures (some tasks succeed, some fail)?
- What's the performance overhead of parallelization?

---

### Phase 2 Evaluation Criteria

For each framework that passes Phase 1, evaluate Phase 2 scenarios:

| Criterion | Scoring | Notes |
|-----------|---------|-------|
| Conditional Routing Syntax | 1-5 | How clean is the routing code? |
| Error Handling Built-ins | 1-5 | Does framework provide retry/fallback? |
| Loop/Cycle Support | 1-5 | How natural are bidirectional flows? |
| Adding New Agents | 1-5 | How much refactoring needed? |
| Parallel Execution | 1-5 | Native support or manual? |
| State Management | 1-5 | How is context preserved across agents? |
| Debugging/Tracing | 1-5 | Can you trace the full flow? |

**Minimum Score to Proceed:** 25/35 (70%)

---

## Phase 3: Mini-Prototype (Top 2 Frameworks)

**Time Estimate:** 3-5 days per framework

**Purpose:** Build a small version of your actual target architecture.

**Pre-requisite:** Pass Phase 2 evaluation with score ≥ 25/35

### Mini-Prototype Specification

Build a 5-agent system that mirrors the actual Nemo architecture:

```
                                          ┌──────────────────┐
                                          │  Manual Review   │
                                          │  Queue (output)  │
                                          └────────▲─────────┘
                                                   │
┌─────────────┐    ┌─────────────┐    ┌───────────┴───────────┐    ┌─────────────┐
│  Guardrails │ → │   Router    │ → │   Weather + Fraud     │ → │  Eligibility │
│  Validator  │    │   Agent     │    │   (parallel check)    │    │    Agent    │
└─────────────┘    └─────────────┘    └───────────────────────┘    └──────┬──────┘
       │                  │                      │                         │
       ▼                  ▼                      ▼                         ▼
   [REJECT if         [Route based         [Error → Manual]           [Final Decision]
    malicious]         on CAT#]                                              │
                                                                             ▼
                                                                    ┌─────────────┐
                                                                    │   Auditor   │
                                                                    │   Agent     │
                                                                    └─────────────┘
```

### Agents to Implement

#### 1. Guardrails Validator
- Checks for prompt injection, malicious content
- Can REJECT immediately (short-circuit)
- Tools: None (pattern matching / LLM reasoning)

#### 2. Router Agent
- Analyzes claim data
- Routes based on: CatastropheNumber present, claim type, policy status
- Tools: None (routing logic)

#### 3. Weather Agent (with parallel Fraud check)
- Fetches BOM weather data
- Runs in parallel with basic fraud indicators
- Tools: geocode_location, fetch_bom_weather

#### 4. Eligibility Agent
- Receives weather + fraud results
- Makes CAT determination
- Can request clarification from Weather Agent (loop)
- Tools: None (reasoning)

#### 5. Auditor Agent
- Compiles final audit report
- Formats output for ACMS diary
- Tools: write_audit_log (mock)

### Advanced Scenarios to Test

#### Human-in-the-Loop
```python
# Test: Pause execution for human approval on high-value claims
if claim_amount > 1000:
    await request_human_approval(claim_id, summary)
    # Resume after approval received
```

#### Dynamic Routing
```python
# Test: Route different claim types to different agent paths
claim_type = detect_claim_type(claim_data)  # "food_spoilage", "storm_damage", "theft"

routing_map = {
    "food_spoilage": [weather_agent, eligibility_agent],
    "storm_damage": [weather_agent, building_assessment_agent, eligibility_agent],
    "theft": [police_report_agent, eligibility_agent]
}
```

#### Long-Running State
```python
# Test: Resume processing after external dependency
# Simulate: Weather API is down, claim is queued
# Later: API is back, claim processing resumes from saved state

saved_state = {
    "claim_id": "12345",
    "completed_agents": ["guardrails", "router"],
    "pending_agent": "weather",
    "context": {...}
}

# Resume from saved state
result = await resume_claim_processing(saved_state)
```

### Phase 3 Evaluation Criteria

| Criterion | Scoring | Notes |
|-----------|---------|-------|
| 5+ Agent Coordination | 1-5 | Complexity management |
| Human-in-the-Loop | 1-5 | Pause/resume capability |
| Dynamic Routing | 1-5 | Runtime path selection |
| State Persistence | 1-5 | Save/resume capability |
| Observability | 1-5 | Tracing, logging at scale |
| Error Propagation | 1-5 | How errors surface up the chain |
| Code Organization | 1-5 | Maintainability at scale |
| Testing Strategy | 1-5 | How to unit test agents |
| Documentation | 1-5 | Can team understand it? |
| Azure Integration | 1-5 | Fit with existing infrastructure |

**Minimum Score to Recommend:** 40/50 (80%)

---

## Phase Summary

| Phase | Frameworks | Time | Purpose |
|-------|------------|------|---------|
| **Phase 1** | All 10 | 1-2 days each | Quick filter - eliminate poor fits |
| **Phase 2** | Top 3-4 | 2-3 days each | Complex scenarios - test real capabilities |
| **Phase 3** | Top 2 | 3-5 days each | Mini-prototype - validate production readiness |

### Decision Gates

**After Phase 1:**
- Eliminate frameworks scoring < 60% on basic criteria
- Select top 3-4 frameworks for Phase 2

**After Phase 2:**
- Eliminate frameworks scoring < 70% on complex scenarios
- Select top 2 frameworks for Phase 3

**After Phase 3:**
- Select final framework (or hybrid approach)
- Document migration plan from current AutoGen implementation

---

## Questions to Answer After Evaluation

## Questions to Answer After Evaluation

### Phase 1: Single Agent Experience
1. Which framework had the cleanest tool definition syntax?
2. Which framework had the best error handling out of the box?
3. Which framework would be easiest for the team to learn?
4. Which framework's documentation was most helpful?
5. Which framework had the most intuitive debugging experience?

### Phase 1: Basic Multi-Agent Experience
6. Which framework made agent-to-agent handoffs most intuitive?
7. Which framework handled state passing between agents best?
8. How readable is the orchestration code in each framework?

### Phase 2: Complex Scenario Questions

**Conditional Routing (Scenario 2.1):**
9. How natural is the conditional routing syntax?
10. Can you easily add more routes at runtime?
11. How is routing state preserved for debugging?

**Error Handling (Scenario 2.2):**
12. Does the framework have built-in retry mechanisms?
13. How do you distinguish transient vs permanent errors?
14. Can you preserve context when routing to fallback?

**Clarification Loops (Scenario 2.3):**
15. How does the framework handle bidirectional communication?
16. Can you limit the number of loop iterations?
17. Is the conversation history preserved correctly?

**Adding Agents (Scenario 2.4):**
18. How easy was it to add a third agent to the flow?
19. Can you conditionally skip agents based on previous results?
20. How do you pass accumulated context to later agents?

**Parallel Execution (Scenario 2.5):**
21. Does the framework have native parallel execution support?
22. How are parallel task results merged?
23. How do you handle partial failures in parallel tasks?

### Phase 3: Production Readiness Questions

**Scale & Complexity:**
24. Which framework scales best to 10+ agents?
25. How well does the framework handle complex conditional flows?
26. Can you easily visualize the full agent flow?

**Enterprise Requirements:**
27. Which framework best supports our Azure infrastructure?
28. How does the framework handle human-in-the-loop scenarios?
29. Can you save and resume agent execution state?

**Maintainability:**
30. Which framework would make testing and debugging easiest?
31. Which framework has the best observability (logging, tracing)?
32. How easy is it to unit test individual agents?

### Strategic Fit Questions
33. Which framework aligns best with expanding to multiple claim types (storm, theft, water damage)?
34. Which framework would allow non-technical team members to understand the agent flow?
35. Which framework has the most active community and is likely to be maintained long-term?
36. What is the migration effort from current AutoGen implementation?
37. Does the framework support a hybrid approach (using multiple frameworks for different purposes)?

---

## Next Steps

### Phase 1 Execution (Weeks 1-2)

1. **Implement basic example** across all 10 frameworks
2. **Document findings** using the output format template
3. **Score each framework** on Phase 1 criteria
4. **Eliminate poor fits** (score < 60%)
5. **Select top 3-4** for Phase 2

### Phase 2 Execution (Weeks 3-4)

6. **Implement complex scenarios** (2.1-2.5) for selected frameworks
7. **Test error handling** and edge cases thoroughly
8. **Evaluate orchestration patterns** for each framework
9. **Score on Phase 2 criteria** (minimum 70% to proceed)
10. **Select top 2** for Phase 3

### Phase 3 Execution (Weeks 5-6)

11. **Build mini-prototype** matching Nemo architecture
12. **Test advanced scenarios** (human-in-loop, dynamic routing, state persistence)
13. **Evaluate Azure integration** and deployment patterns
14. **Score on Phase 3 criteria** (minimum 80% to recommend)

### Final Decision (Week 7)

15. **Compare final candidates** side-by-side
16. **Team review session** - Demo both frameworks to the team
17. **Decision meeting** - Select framework and document rationale
18. **Migration planning** - Create plan to migrate from current AutoGen implementation

### Deliverables

| Phase | Deliverable | Due |
|-------|-------------|-----|
| Phase 1 | Framework comparison matrix with scores | Week 2 |
| Phase 2 | Complex scenario test results | Week 4 |
| Phase 3 | Mini-prototype demos and evaluation | Week 6 |
| Final | Framework recommendation document | Week 7 |
| Final | Migration plan (if switching frameworks) | Week 8 |
