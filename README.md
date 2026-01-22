# AI Agent Framework Comparison Guide

A comprehensive evaluation of **11 AI agent frameworks** for building production systems, with a focus on insurance and regulated industries.

## Quick Decision Guide

```
                    ┌─────────────────────────────────┐
                    │   Which framework should I use?  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Need type-safe structured    │
                    │  outputs? (insurance/finance) │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │ YES                 │                 NO  │
              ▼                     │                     ▼
    ┌─────────────────┐             │         ┌─────────────────┐
    │  PYDANTIC AI    │             │         │  Using Claude   │
    │  ⭐ Recommended  │             │         │  models only?   │
    └─────────────────┘             │         └────────┬────────┘
                                    │                  │
                                    │     ┌────────────┼────────────┐
                                    │     │ YES        │        NO  │
                                    │     ▼            │            ▼
                                    │ ┌─────────┐      │    ┌─────────────┐
                                    │ │ANTHROPIC│      │    │OpenAI only? │
                                    │ └─────────┘      │    └──────┬──────┘
                                    │                  │           │
                                    │                  │    ┌──────┼──────┐
                                    │                  │    │YES   │   NO │
                                    │                  │    ▼      │      ▼
                                    │                  │ ┌──────┐  │  ┌──────────┐
                                    │                  │ │OPENAI│  │  │ See full │
                                    │                  │ │AGENTS│  │  │ guide... │
                                    │                  │ └──────┘  │  └──────────┘
                                    │                  │           │
                                    └──────────────────┴───────────┘
```

**[See Interactive Decision Wizard →](decision_flowchart.html)** (Open in browser)

## Frameworks Covered

| Framework | Best For | Install |
|-----------|----------|---------|
| **Pydantic AI** ⭐ | Type safety, regulated industries | `pip install pydantic-ai` |
| **LangGraph** | State machines, debuggable workflows | `pip install langgraph` |
| **OpenAI Agents SDK** | Simple handoffs, OpenAI-only | `pip install openai` |
| **AutoGen** | Group chat, collaborative agents | `pip install autogen-agentchat` |
| **CrewAI** | Role-based teams | `pip install crewai` |
| **Anthropic** | Claude models | `pip install anthropic` |
| **Haystack** | NLP pipelines, RAG | `pip install haystack-ai` |
| **Semantic Kernel** | Azure/Microsoft | `pip install semantic-kernel` |
| **Smolagents** | Fast prototyping | `pip install smolagents` |
| **LlamaIndex** | Document retrieval | `pip install llama-index` |
| **DSPy** | Prompt optimization | `pip install dspy` |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/srepho/dory.git
cd dory

# Create environment
conda create -n dory python=3.12 -y
conda activate dory

# Install base dependencies
pip install httpx beautifulsoup4 python-dotenv

# Set up API keys
echo 'OPENAI_API_KEY="your-key"' > .env
echo 'ANTHROPIC_API_KEY="your-key"' >> .env  # Optional

# Run any demo
python pydantic_ai_demo.py
```

## Repository Structure

```
dory/
├── blog_post.md              # Full comparison guide (start here!)
├── decision_flowchart.html   # Interactive decision wizard
├── decision_flowchart.py     # CLI decision tool
│
├── *_demo.py                 # Working demos for each framework
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
├── tutorials/                # Jupyter notebook tutorials
│   ├── 01_pydantic_ai_tutorial.ipynb
│   ├── 02_langgraph_tutorial.ipynb
│   ├── ... (11 tutorials total)
│   └── 11_dspy_tutorial.ipynb
│
├── shared_utils.py           # Common utilities
└── demo_config.py            # Shared configuration
```

## The Use Case: Insurance Weather Verification

All frameworks implement the same use case for fair comparison:

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│   User       │────▶│  Weather Agent      │────▶│  Eligibility Agent   │
│   Request    │     │  (has tools)        │     │  (LLM reasoning)     │
└──────────────┘     │                     │     │                      │
                     │  1. Geocode address │     │  Apply business      │
  Location: Brisbane │  2. Fetch BOM data  │     │  rules:              │
  Date: 2025-03-07   │  3. Compile report  │     │  • APPROVED          │
                     └─────────────────────┘     │  • REVIEW            │
                                                 │  • DENIED            │
                                                 └──────────────────────┘
```

**CAT Event Rules:**
- **APPROVED**: Both thunderstorms AND strong winds detected
- **REVIEW**: Only one severe weather type detected
- **DENIED**: No severe weather detected

## Tutorials

Each tutorial covers:
1. Framework installation & setup
2. Core concepts explained
3. Building the Weather Agent
4. Building the Eligibility Agent
5. DSPy integration for prompt optimization
6. MLFlow integration for experiment tracking

| Tutorial | Framework | Notebook |
|----------|-----------|----------|
| 01 | Pydantic AI | [01_pydantic_ai_tutorial.ipynb](tutorials/01_pydantic_ai_tutorial.ipynb) |
| 02 | LangGraph | [02_langgraph_tutorial.ipynb](tutorials/02_langgraph_tutorial.ipynb) |
| 03 | AutoGen | [03_autogen_tutorial.ipynb](tutorials/03_autogen_tutorial.ipynb) |
| 04 | CrewAI | [04_crewai_tutorial.ipynb](tutorials/04_crewai_tutorial.ipynb) |
| 05 | OpenAI Agents | [05_openai_agents_tutorial.ipynb](tutorials/05_openai_agents_tutorial.ipynb) |
| 06 | Anthropic | [06_anthropic_tutorial.ipynb](tutorials/06_anthropic_tutorial.ipynb) |
| 07 | Haystack | [07_haystack_tutorial.ipynb](tutorials/07_haystack_tutorial.ipynb) |
| 08 | Semantic Kernel | [08_semantic_kernel_tutorial.ipynb](tutorials/08_semantic_kernel_tutorial.ipynb) |
| 09 | Smolagents | [09_smolagents_tutorial.ipynb](tutorials/09_smolagents_tutorial.ipynb) |
| 10 | LlamaIndex | [10_llamaindex_tutorial.ipynb](tutorials/10_llamaindex_tutorial.ipynb) |
| 11 | DSPy | [11_dspy_tutorial.ipynb](tutorials/11_dspy_tutorial.ipynb) |

## Decision Tools

### Interactive HTML Wizard
```bash
# Open in browser
open decision_flowchart.html
```

### CLI Decision Tool
```bash
# Interactive wizard
python decision_flowchart.py

# Print quick reference
python decision_flowchart.py --table
```

## Key Recommendations

| Your Situation | Recommendation |
|----------------|----------------|
| New team + regulated industry | **Pydantic AI** - Type safety prevents data format errors |
| Need debuggable workflows | **LangGraph** - Explicit state machines + LangSmith |
| OpenAI only, want simplicity | **OpenAI Agents SDK** - Minimal code |
| Claude models only | **Anthropic SDK** - Native tool support |
| Document retrieval (RAG) | **LlamaIndex** - Built for knowledge bases |
| Azure/Microsoft shop | **Semantic Kernel** - Enterprise integration |
| Want to optimize prompts | **DSPy** - Works with any framework |

## Full Documentation

📖 **[Read the full comparison guide →](blog_post.md)**

The guide includes:
- Detailed code examples for each framework
- Common pitfalls and how to avoid them
- Production readiness assessment
- Lines of code comparison
- Model flexibility analysis

## Contributing

Contributions welcome! Please:
1. Test your changes with actual API calls
2. Follow the existing code style
3. Update relevant documentation

## License

MIT License - feel free to use this for your own evaluations.

---

*Built with Claude Opus 4.5 for an insurance company's agent framework evaluation.*
