# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conda Environment

**IMPORTANT: Always use conda to manage environments. Never use pip install directly in base.**

```bash
# Create environment (Python 3.12)
conda create -n dory python=3.12 -y
conda activate dory

# Install dependencies
pip install httpx beautifulsoup4 python-dotenv
pip install langchain langchain-openai langgraph  # LangGraph
pip install pydantic-ai  # Pydantic AI
pip install crewai crewai-tools  # CrewAI
pip install autogen-agentchat autogen-ext[openai]  # AutoGen

# Verify environment
which python  # Should show conda env path
```

**Before running any code:**
```bash
conda activate dory
```

## Project Overview

Dory is an **AI agent framework evaluation guide** - a comprehensive specification document for evaluating and comparing 10 AI agent frameworks for building multi-agent systems. The primary document is `agent_framework_evaluation_updated.md` (~2,900 lines).

**Domain**: Insurance claims processing - verifying catastrophic (CAT) weather events for insurance eligibility in Australia.

**Note**: This is a specification/planning project, not an executable codebase. There are no build/test/lint commands.

## Example Agents Specified

Two interconnected agents to implement across all frameworks:

1. **Weather Verification Agent** (has tools)
   - Geocodes location using Nominatim API
   - Queries Australian Bureau of Meteorology (BOM) for weather observations
   - Returns: location, coordinates, weather events (thunderstorms/strong wind), severity confirmation

2. **Claims Eligibility Agent** (LLM reasoning only)
   - Applies business rules for CAT event classification
   - No tools - pure reasoning
   - Returns: eligibility decision (APPROVED/REVIEW/DENIED)

## Frameworks Evaluated

| Framework | Install | Multi-Agent Pattern |
|-----------|---------|---------------------|
| AutoGen 0.4+ | `pip install autogen-agentchat autogen-ext` | RoundRobinGroupChat, SelectorGroupChat |
| LangChain + LangGraph | `pip install langchain langchain-openai langgraph` | StateGraph with conditional edges |
| Pydantic AI | `pip install pydantic-ai` | Manual sequential pipeline |
| OpenAI Agents SDK | `pip install openai` | Explicit handoff functions |
| Anthropic Claude | `pip install anthropic` | Manual orchestration |
| Haystack | `pip install haystack-ai` | Custom pipeline components |
| Azure Semantic Kernel | `pip install semantic-kernel` | Plugins + semantic functions |
| CrewAI | `pip install crewai crewai-tools` | Task dependencies |
| Smolagents | `pip install smolagents` | ManagedAgent wrapper |
| LlamaIndex | `pip install llama-index llama-index-agent-openai` | Sequential agents |

## APIs Used

- **Nominatim**: `https://nominatim.openstreetmap.org/search` - Geocoding (requires User-Agent header)
- **BOM Storms**: `https://reg.bom.gov.au/cgi-bin/climate/storms/get_storms.py` - Weather observations (returns HTML, needs parsing)

## Evaluation Criteria

13 dimensions: Setup Complexity, Tool Definition, Orchestration Options, Type Safety, Error Handling, Observability, Multi-Agent Support, Streaming, Model Flexibility, Production Readiness, Documentation Quality, Community/Maintenance, Cost/Performance

## Multi-Agent Orchestration Patterns

1. **Sequential Handoff**: User → Agent1 → [results] → Agent2 → Response
2. **Supervisor/Router**: Router decides which agents to invoke
3. **Collaborative Chat**: Agents in shared conversation context
4. **State Machine**: Explicit states with conditional transitions

## Test Cases

```
1. Brisbane, QLD, 4000 on 2025-03-07
2. Mcdowall, QLD, 4053 on 2025-03-07
3. Sydney, NSW, 2000 on 2025-03-07
4. Perth, WA, 6000 on 2025-01-15
```

## Business Rules (for Claims Eligibility Agent)

- **APPROVED**: BOTH thunderstorms AND strong wind observed + valid Australian location + valid date
- **REVIEW**: Only ONE weather type observed
- **DENIED**: Neither observed or invalid data
- Australia bounds: lat -44 to -10, lon 112 to 154
- Date must be within 90 days, not in future

## Implementation Files to Create

When implementing, create:
- `shared_utils.py` - geocode_address(), fetch_bom_observations(), is_severe_weather()
- `{framework}_weather_agent.py` - One file per framework with both agents
