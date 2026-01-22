"""
Shared configuration for AI Agent Framework demos.

This module provides:
- API key loading from .env file
- Model configuration (z.ai primary, OpenAI fallback)
- Test case data
- Common imports setup

All demo files import from here to ensure consistent configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API Configuration
# ============================================================================

# Use z.ai API as primary (cheaper for demos) - OpenAI-compatible endpoint
# Falls back to OpenAI if z.ai not configured
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_API_BASE = os.getenv("ZHIPU_API_BASE", "https://api.z.ai/api/paas/v4")

# Anthropic API (for Claude demos)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Azure OpenAI (for Semantic Kernel)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

# Hugging Face (for Smolagents)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

# Configure OpenAI-compatible clients to use z.ai if available
if ZHIPU_API_KEY:
    os.environ["OPENAI_API_KEY"] = ZHIPU_API_KEY
    os.environ["OPENAI_BASE_URL"] = ZHIPU_API_BASE.replace("/chat/completions", "")
    USE_ZAI = True
    # z.ai models: glm-4.5, glm-4.5-air, glm-4.6, glm-4.7
    DEFAULT_MODEL = "glm-4.5-air"  # Fast and cheap for demos
    print(f"✓ Using z.ai API: {ZHIPU_API_KEY[:12]}...")
elif os.getenv("OPENAI_API_KEY"):
    USE_ZAI = False
    DEFAULT_MODEL = "gpt-4o-mini"
    print(f"✓ Using OpenAI API: {os.getenv('OPENAI_API_KEY')[:8]}...")
else:
    USE_ZAI = False
    DEFAULT_MODEL = "gpt-4o-mini"
    print("⚠ Warning: No API key found. Set ZHIPU_API_KEY or OPENAI_API_KEY in .env")


# ============================================================================
# Test Case Data
# ============================================================================

# Default test case - Brisbane with known weather events
TEST_CITY = "Brisbane"
TEST_STATE = "QLD"
TEST_POSTCODE = "4000"
TEST_DATE = "2025-03-07"

# Additional test cases from CLAUDE.md
TEST_CASES = [
    {"city": "Brisbane", "state": "QLD", "postcode": "4000", "date": "2025-03-07"},
    {"city": "Mcdowall", "state": "QLD", "postcode": "4053", "date": "2025-03-07"},
    {"city": "Sydney", "state": "NSW", "postcode": "2000", "date": "2025-03-07"},
    {"city": "Perth", "state": "WA", "postcode": "6000", "date": "2025-01-15"},
]


# ============================================================================
# System Prompts
# ============================================================================

WEATHER_AGENT_SYSTEM_PROMPT = """You are a Weather Verification Agent. Your job is to verify severe weather events for insurance claims.

STEPS:
1. Geocode the provided location (city, state, postcode) to get latitude/longitude coordinates
2. Fetch weather observations from the Bureau of Meteorology (BOM) for the given date
3. Report your findings in a structured format

You have access to these tools:
- geocode_location: Convert address to coordinates
- get_bom_weather: Fetch BOM weather observations

Always use your tools - never make up data. Report actual observations."""


ELIGIBILITY_AGENT_SYSTEM_PROMPT = """You are a Claims Eligibility Agent. You evaluate weather verification reports and determine CAT (catastrophic) event eligibility.

RULES:
- BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
- Only ONE weather type "Observed" = POSSIBLE CAT → REVIEW
- Neither "Observed" = NOT CAT → DENIED

VALIDATION:
- Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
- Date must be valid and not in the future

Respond with a JSON decision: {cat_event_status, eligibility_decision, confidence, reasoning}"""


# ============================================================================
# Helper Functions
# ============================================================================

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Using z.ai: {USE_ZAI}")
    print(f"Test Case: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
