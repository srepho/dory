# Session Notes - Dory AI Agent Framework Guide

## Last Updated: 2026-01-30 (Session 5)

## Project Status: READY FOR DECISION ✓

The Dory repository is a comprehensive AI agent framework comparison guide for an insurance company team evaluating frameworks for building production multi-agent systems.

**GitHub:** https://github.com/srepho/dory

---

## Repository Structure

```
dory/
├── README.md                    # Project overview with quick decision guide
├── blog_post.md                 # Full comparison guide (~150KB)
├── framework_comparison.md      # 5-framework comparison (~1800 lines, Mermaid charts)
├── framework_comparison_print.md # Print version with ASCII charts for PDF
├── framework_comparison.pdf     # PDF export
├── slides.md                    # NEW: 35-slide presentation deck
├── decision_flowchart.html      # Interactive web wizard (Cytoscape.js)
├── decision_flowchart.py        # CLI decision tool
├── shared_utils.py              # Geocoding + BOM weather utilities
├── demo_config.py               # Shared test configuration
├── CLAUDE.md                    # Project instructions
├── .gitignore                   # Excludes .env, pycache, etc.
│
├── *_demo.py (13 files)         # Individual framework demos
│   ├── pydantic_ai_demo.py
│   ├── langgraph_demo.py
│   ├── autogen_demo.py          # Old AutoGen 0.4+
│   ├── ms_agent_framework_demo.py  # NEW: Microsoft Agent Framework
│   ├── claude_agent_sdk_demo.py    # NEW: Claude Agent SDK
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

### Session 4 (Framework Comparison + New Demos)
- Created `framework_comparison.md` - comprehensive comparison of 5 frameworks:
  - Haystack, LangGraph, Pydantic AI, MS Agent Framework, Claude Agent SDK
- Added sections:
  - Philosophy & mental models for each framework
  - Observability deep dive
  - Failure modes & error handling
  - Complex workflow patterns
  - Real-world implementation testing
  - LLM Learnability benchmark (from fwork-learnability repo)
- Created `slides.md` - 35-slide presentation deck
- Created new demo implementations:
  - `ms_agent_framework_demo.py` - Microsoft Agent Framework (new unified framework)
  - `claude_agent_sdk_demo.py` - Claude Agent SDK with MCP tools
- Note: New frameworks (MS Agent Framework, Claude Agent SDK) are very new (Apr/Jun 2025)
  and may have API changes. Demos include fallback modes.

### Session 5 (Comprehensive Tables + Updated Learnability Data)
- Added **Comprehensive Comparison Table** section with 12 sub-tables:
  - Core Framework Attributes (including "Stable API Since" for contamination context)
  - Code Verbosity (actual LoC from demo files: 186-363 lines)
  - Tool Definition & Type Safety
  - Multi-Agent Patterns
  - Observability
  - Error Handling & Recovery
  - Workflow Capabilities
  - Production Readiness
  - Framework Overhead
  - LLM Learnability
  - Philosophy Summary
  - Overall Scores (subjective star ratings)

- Updated **LLM Learnability Benchmark** with new experimental data from fwork-learnability:
  - Experiment 1 (scraped docs): Pydantic AI 92%, Haystack 75%
  - Experiment 2 (curated docs): Haystack 100%, LangGraph 100%, OpenAI Agents 100%, Anthropic Agents 92%
  - Added contamination analysis (all frameworks 100% success with zero docs)
  - Key insight: Haystack improved 75% → 100% with curated docs vs scraped marketing pages
  - LangGraph consistently slowest (4.7-5.4 avg turns)

- Fixed DeepSeek V3 training cutoff to **July 2024** (not Dec 2024)

- Created `framework_comparison_print.md` with ASCII charts for PDF export
- Regenerated `framework_comparison.pdf`

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

## Recommended Next Steps

### Immediate (Decision Support)
- [ ] **Pick a framework** - Data supports decision:
  - **Pydantic AI** → Type safety, code elegance, regulated industries (RECOMMENDED)
  - **Haystack** → RAG/document processing primary use case
  - **LangGraph** → Complex stateful workflows with checkpointing
- [ ] **Create one-page executive summary** - Full doc is ~1700 lines, need single-page for stakeholders

### Short-term (Validation)
- [ ] **Build proof-of-concept** with chosen framework using real insurance scenario
- [ ] **Run Tier 2/3 learnability tests** - Current benchmark only tests classification, not tool use or agent orchestration
- [ ] **Re-test Pydantic AI with curated docs** - Scored 92% with scraped docs, likely 100% with curated

### Medium-term (Production Readiness)
- [ ] **Cost/latency benchmarking** - Token counting and timing across frameworks
- [ ] **Set up observability** - Logfire (Pydantic AI), LangSmith (LangGraph), or OpenTelemetry
- [ ] **Security review** - Data handling, API key management, audit logging for insurance context
- [ ] **Deployment examples** - Docker, cloud deployment patterns

### Backlog
- [ ] Add more test cases beyond Brisbane
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
