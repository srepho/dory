# Session Notes - Dory AI Agent Framework Guide

## Last Updated: 2026-01-23

## Project Status: COMPLETE ✓

The Dory repository is a comprehensive AI agent framework comparison guide for an insurance company team evaluating frameworks for building production multi-agent systems.

**GitHub:** https://github.com/srepho/dory

---

## Repository Structure

```
dory/
├── README.md                    # Project overview with quick decision guide
├── blog_post.md                 # Full comparison guide (~150KB)
├── decision_flowchart.html      # Interactive web wizard (Cytoscape.js)
├── decision_flowchart.py        # CLI decision tool
├── shared_utils.py              # Geocoding + BOM weather utilities
├── demo_config.py               # Shared test configuration
├── CLAUDE.md                    # Project instructions
├── .gitignore                   # Excludes .env, pycache, etc.
│
├── *_demo.py (11 files)         # Individual framework demos
│   ├── pydantic_ai_demo.py
│   ├── langgraph_demo.py
│   ├── autogen_demo.py
│   ├── crewai_demo.py
│   ├── openai_agents_demo.py
│   ├── anthropic_demo.py
│   ├── haystack_demo.py
│   ├── semantic_kernel_demo.py
│   ├── smolagents_demo.py
│   ├── llamaindex_demo.py
│   └── dspy_demo.py
│
└── tutorials/ (11 notebooks)    # Jupyter tutorials with DSPy + MLFlow
    ├── 01_pydantic_ai_tutorial.ipynb
    ├── 02_langgraph_tutorial.ipynb
    ├── ... (03-10)
    └── 11_dspy_tutorial.ipynb
```

---

## What Was Completed

### Session 1 (Initial Build)
- Created 11 demo files for each framework
- Created blog_post.md comprehensive comparison
- Created shared_utils.py with geocoding + BOM API

### Session 2 (Tutorials + Polish)
- Created all 11 Jupyter tutorial notebooks
- Created decision_flowchart.html (interactive web wizard)
- Created decision_flowchart.py (CLI tool)
- Created README.md
- Removed "progression path" advice (team will pick ONE framework)
- Pushed to GitHub

### Session 3 (Cleanup + Testing)
- Removed redundant files:
  - `agent_framework_evaluation_updated.md` (superseded by blog_post.md)
  - `agent_frameworks_demo.py` (superseded by individual demos)
  - `feedback.md` (addressed)
- Verified all components working:
  - ✓ Core dependencies (httpx, bs4, dotenv)
  - ✓ decision_flowchart.py CLI
  - ✓ shared_utils.py (geocoding + BOM API)
  - ✓ anthropic_demo.py end-to-end
  - ✓ dspy_demo.py end-to-end
  - ✓ All 11 tutorials valid JSON + Python syntax

---

## Test Results

**APIs verified working:**
- Nominatim geocoding: ✓ (Brisbane → -27.47, 153.02)
- BOM weather: ✓ (returns thunderstorm/wind observations)

**Installed frameworks on this machine:**
- ✓ openai, anthropic, dspy
- ✗ pydantic-ai, langchain, langgraph, crewai, autogen, haystack, semantic-kernel, smolagents, llama-index (need pip install)

**API keys configured in .env:**
- OPENAI_API_KEY: set
- ANTHROPIC_API_KEY: set

---

## Key Design Decisions

1. **Single framework focus**: Removed "progression path" advice. Team will pick ONE tool and learn it well.

2. **Use case**: Insurance weather verification (CAT events)
   - Agent 1: Weather Verification (geocoding + BOM API tools)
   - Agent 2: Claims Eligibility (pure LLM reasoning)
   - Rules: APPROVED (both storm+wind), REVIEW (one), DENIED (neither)

3. **Recommended framework**: Pydantic AI for type safety in regulated industries

---

## Potential Future Work

- [ ] Add more test cases beyond Brisbane
- [ ] Add comparison benchmarks (latency, token usage, cost)
- [ ] Add deployment examples (Docker, cloud)
- [ ] Record video walkthrough of decision wizard
- [ ] Add framework version pinning for reproducibility

---

## Quick Commands

```bash
# Test decision tool
python decision_flowchart.py --table

# Test shared utilities
python -c "from shared_utils import geocode_address; print(geocode_address('Brisbane', 'QLD', '4000'))"

# Run a demo
python anthropic_demo.py

# Open interactive wizard
open decision_flowchart.html
```
