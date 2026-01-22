"""
DSPy Demo - Weather Verification Multi-Agent System

Stanford NLP's framework for programming—not prompting—language models.

Key features:
- Declarative signatures define input/output
- Modules (ChainOfThought, ReAct) provide reasoning patterns
- Optimizers (GEPA, MIPROv2) automatically improve prompts
- MLFlow integration for experiment tracking

Philosophy:
DSPy treats prompts as optimizable programs. Instead of hand-crafting prompts,
you define what you want (signatures) and let optimizers find the best prompts.

Install:
    pip install dspy mlflow

Usage:
    conda activate dory
    python dspy_demo.py
"""

import os
import json

# Import shared configuration
from demo_config import (
    DEFAULT_MODEL, USE_ZAI, TEST_CITY, TEST_STATE, TEST_POSTCODE, TEST_DATE
)

# Import shared utilities for tool implementations
from shared_utils import geocode_address, fetch_bom_observations


def run_dspy_demo():
    """Run the DSPy multi-agent demo."""
    try:
        import dspy
    except ImportError:
        print("DSPy not installed. Run: pip install dspy")
        return None

    print(f"\n{'='*60}")
    print("DSPy Demo")
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Test: {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}")
    print()

    # --- Configure LM ---
    # DSPy uses its own LM configuration
    if USE_ZAI:
        lm = dspy.LM(
            model=f"openai/{DEFAULT_MODEL}",
            api_base=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_tokens=2000
        )
    else:
        lm = dspy.LM(
            model=f"openai/{DEFAULT_MODEL}",
            max_tokens=2000
        )

    dspy.configure(lm=lm)

    # --- Tool Definitions ---
    # DSPy tools are regular Python functions with docstrings
    def geocode_location(city: str, state: str, postcode: str) -> str:
        """Convert an Australian address to latitude/longitude coordinates.

        Args:
            city: City name (e.g., "Brisbane")
            state: Australian state code (e.g., "QLD")
            postcode: Postcode (e.g., "4000")

        Returns:
            JSON string with latitude, longitude, and display_name
        """
        print(f"  → Geocoding: {city}, {state}, {postcode}")
        result = geocode_address(city, state, postcode)
        print(f"    → {result}")
        return json.dumps(result)

    def get_bom_weather(lat: float, lon: float, date: str, state: str) -> str:
        """Fetch weather observations from Australian Bureau of Meteorology.

        Args:
            lat: Latitude (e.g., -27.5)
            lon: Longitude (e.g., 153.0)
            date: Date in YYYY-MM-DD format
            state: Australian state code (e.g., "QLD")

        Returns:
            JSON string with thunderstorms and strong_wind observations
        """
        print(f"  → Fetching BOM weather: ({lat}, {lon}) on {date}")
        result = fetch_bom_observations(lat, lon, date, state)
        print(f"    → {result}")
        return json.dumps(result)

    # --- Weather Agent using ReAct ---
    print("Creating Weather Agent (ReAct)...")

    # ReAct agent with tools
    weather_agent = dspy.ReAct(
        signature="task -> weather_report",
        tools=[geocode_location, get_bom_weather],
        max_iters=5
    )

    # --- Eligibility Agent using ChainOfThought ---
    print("Creating Eligibility Agent (ChainOfThought)...")

    # Define signature for eligibility determination
    class EligibilitySignature(dspy.Signature):
        """Determine CAT event eligibility based on weather verification.

        Business Rules:
        - BOTH thunderstorms AND strong wind "Observed" = CONFIRMED CAT → APPROVED
        - Only ONE weather type "Observed" = POSSIBLE CAT → REVIEW
        - Neither "Observed" = NOT CAT → DENIED

        Validation:
        - Coordinates must be in Australia (-44 to -10 lat, 112 to 154 lon)
        """
        weather_report: str = dspy.InputField(desc="Weather verification report with location, coordinates, and observations")
        cat_event_status: str = dspy.OutputField(desc="CONFIRMED, POSSIBLE, or NOT_CAT")
        eligibility_decision: str = dspy.OutputField(desc="APPROVED, REVIEW, or DENIED")
        confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")
        reasoning: str = dspy.OutputField(desc="Explanation of the decision")

    eligibility_agent = dspy.ChainOfThought(EligibilitySignature)

    # --- Run Sequential Pipeline ---
    print("\nRunning Weather Agent...")
    weather_task = f"""Verify weather for {TEST_CITY}, {TEST_STATE}, {TEST_POSTCODE} on {TEST_DATE}.

    Steps:
    1. Use geocode_location to get coordinates for the address
    2. Use get_bom_weather to fetch weather observations for that date
    3. Report your findings including location, coordinates, thunderstorms, and strong_wind observations"""

    weather_result = weather_agent(task=weather_task)
    weather_report = weather_result.weather_report

    print(f"\nWeather Agent Output:\n{weather_report}")

    print("\nRunning Eligibility Agent...")
    eligibility_result = eligibility_agent(weather_report=weather_report)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nWeather Verification:\n{weather_report}")
    print(f"\nEligibility Decision:")
    print(f"  CAT Status: {eligibility_result.cat_event_status}")
    print(f"  Decision: {eligibility_result.eligibility_decision}")
    print(f"  Confidence: {eligibility_result.confidence}")
    print(f"  Reasoning: {eligibility_result.reasoning}")

    return {
        "weather": weather_report,
        "eligibility": {
            "cat_event_status": eligibility_result.cat_event_status,
            "eligibility_decision": eligibility_result.eligibility_decision,
            "confidence": str(eligibility_result.confidence),
            "reasoning": eligibility_result.reasoning
        }
    }


def run_dspy_optimization_demo():
    """
    Demonstrate DSPy's optimization capabilities with GEPA.

    This shows how DSPy can automatically optimize prompts to improve performance.
    GEPA (Genetic-Pareto) is a reflective optimizer that evolves prompts based on
    evaluation feedback.
    """
    try:
        import dspy
    except ImportError:
        print("DSPy not installed. Run: pip install dspy")
        return None

    print(f"\n{'='*60}")
    print("DSPy Optimization Demo (GEPA)")
    print(f"{'='*60}")

    # Configure LM
    if USE_ZAI:
        lm = dspy.LM(
            model=f"openai/{DEFAULT_MODEL}",
            api_base=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_tokens=2000
        )
    else:
        lm = dspy.LM(model=f"openai/{DEFAULT_MODEL}", max_tokens=2000)

    dspy.configure(lm=lm)

    # --- Define a simple CAT eligibility classifier ---
    class CATClassifier(dspy.Signature):
        """Classify insurance claims based on weather observations."""
        thunderstorms: str = dspy.InputField(desc="Thunderstorm observation: 'Observed' or 'No reports'")
        strong_wind: str = dspy.InputField(desc="Strong wind observation: 'Observed' or 'No reports'")
        decision: str = dspy.OutputField(desc="APPROVED, REVIEW, or DENIED")

    # Create the module
    classifier = dspy.ChainOfThought(CATClassifier)

    # --- Training data (examples) ---
    trainset = [
        dspy.Example(
            thunderstorms="Observed",
            strong_wind="Observed",
            decision="APPROVED"
        ).with_inputs("thunderstorms", "strong_wind"),
        dspy.Example(
            thunderstorms="Observed",
            strong_wind="No reports or observations",
            decision="REVIEW"
        ).with_inputs("thunderstorms", "strong_wind"),
        dspy.Example(
            thunderstorms="No reports or observations",
            strong_wind="Observed",
            decision="REVIEW"
        ).with_inputs("thunderstorms", "strong_wind"),
        dspy.Example(
            thunderstorms="No reports or observations",
            strong_wind="No reports or observations",
            decision="DENIED"
        ).with_inputs("thunderstorms", "strong_wind"),
    ]

    # --- Evaluation metric ---
    def accuracy_metric(example, prediction, trace=None):
        """Check if prediction matches expected decision."""
        return prediction.decision.upper() == example.decision.upper()

    # --- Evaluate before optimization ---
    print("\nBefore Optimization:")
    correct = 0
    for ex in trainset:
        pred = classifier(thunderstorms=ex.thunderstorms, strong_wind=ex.strong_wind)
        is_correct = pred.decision.upper() == ex.decision.upper()
        correct += int(is_correct)
        print(f"  Thunder: {ex.thunderstorms[:8]:8} | Wind: {ex.strong_wind[:8]:8} | "
              f"Expected: {ex.decision:8} | Got: {pred.decision:8} | {'✓' if is_correct else '✗'}")
    print(f"  Accuracy: {correct}/{len(trainset)} = {100*correct/len(trainset):.0f}%")

    # --- Optimization with GEPA ---
    # Note: GEPA optimization requires multiple iterations and API calls
    # For a quick demo, we'll show the setup but skip actual optimization
    print("\n--- GEPA Optimizer Setup ---")
    print("""
    # To optimize with GEPA:
    from dspy.teleprompt import GEPA

    optimizer = GEPA(
        metric=accuracy_metric,
        max_iterations=3,      # Number of evolution rounds
        num_candidates=4,      # Candidates per iteration
    )

    optimized_classifier = optimizer.compile(
        classifier,
        trainset=trainset,
        # Optional: provide textual feedback for better optimization
        feedback_fn=lambda ex, pred: "Focus on the business rules: both observed = APPROVED"
    )

    # The optimized classifier will have improved prompts that GEPA discovered
    """)

    # --- Show how to check optimization results ---
    print("--- Inspecting DSPy Program ---")
    print(f"Module type: {type(classifier).__name__}")
    # Access signature through the predict attribute in ChainOfThought
    if hasattr(classifier, 'predict'):
        print(f"Signature: {classifier.predict.signature}")
    else:
        print(f"Signature class: {CATClassifier.__name__}")

    return {"demo": "optimization_setup_complete"}


def run_mlflow_integration_demo():
    """
    Demonstrate DSPy + MLFlow integration for experiment tracking.

    MLFlow provides:
    - Automatic tracing of DSPy module calls
    - Tracking of optimization runs
    - Artifact storage for optimized programs
    """
    try:
        import dspy
        import mlflow
        import mlflow.dspy
    except ImportError as e:
        print(f"Required package not installed: {e}")
        print("Run: pip install dspy mlflow")
        return None

    print(f"\n{'='*60}")
    print("DSPy + MLFlow Integration Demo")
    print(f"{'='*60}")

    # --- Enable MLFlow autologging ---
    print("\n1. Enabling MLFlow autologging...")
    mlflow.dspy.autolog(
        log_compiles=True,           # Log optimization runs
        log_evals=True,              # Log evaluations
        log_traces_from_compile=True # Log traces during compilation
    )

    # --- Set experiment ---
    print("2. Setting MLFlow experiment...")
    mlflow.set_experiment("DSPy-Weather-Verification")

    # --- Configure DSPy ---
    if USE_ZAI:
        lm = dspy.LM(
            model=f"openai/{DEFAULT_MODEL}",
            api_base=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_tokens=1000
        )
    else:
        lm = dspy.LM(model=f"openai/{DEFAULT_MODEL}", max_tokens=1000)

    dspy.configure(lm=lm)

    # --- Run a simple prediction with tracking ---
    print("3. Running prediction with MLFlow tracking...")

    class SimpleWeatherCheck(dspy.Signature):
        """Check if weather conditions indicate a CAT event."""
        conditions: str = dspy.InputField()
        is_cat_event: bool = dspy.OutputField()
        explanation: str = dspy.OutputField()

    checker = dspy.ChainOfThought(SimpleWeatherCheck)

    with mlflow.start_run(run_name="weather-check-demo"):
        # Log parameters
        mlflow.log_param("model", DEFAULT_MODEL)
        mlflow.log_param("test_location", f"{TEST_CITY}, {TEST_STATE}")

        # Run prediction (automatically traced)
        result = checker(conditions="Thunderstorms observed, Strong wind observed")

        # Log results
        mlflow.log_metric("is_cat_event", 1 if result.is_cat_event else 0)

        print(f"\n   Result: is_cat_event={result.is_cat_event}")
        print(f"   Explanation: {result.explanation}")
        print("\n   (Check MLFlow UI for traces: mlflow ui)")

    print("\n--- MLFlow Integration Code ---")
    print("""
    # Full MLFlow integration example:

    import mlflow
    import mlflow.dspy
    import dspy

    # Enable autologging
    mlflow.dspy.autolog(log_compiles=True, log_evals=True)

    # Set tracking server (optional - defaults to local)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("my-dspy-experiment")

    # Your DSPy code runs with automatic tracing
    with mlflow.start_run():
        # Optimization runs are tracked with parent/child hierarchy
        optimized = optimizer.compile(module, trainset=data)

        # Log the optimized program as an artifact
        mlflow.dspy.log_model(optimized, "optimized_model")

    # Later, load the optimized model
    loaded = mlflow.dspy.load_model("runs:/<run_id>/optimized_model")
    """)

    return {"mlflow_demo": "complete"}


def show_cross_framework_optimization():
    """
    Show how DSPy can be used to optimize prompts for OTHER frameworks.

    The key insight: DSPy optimizers like GEPA can improve prompts that you
    then export and use in LangChain, CrewAI, AutoGen, etc.
    """
    print(f"\n{'='*60}")
    print("DSPy for Cross-Framework Prompt Optimization")
    print(f"{'='*60}")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                DSPy AS PROMPT OPTIMIZATION LAYER                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │   1. DEFINE your task in DSPy with signatures                       │
    │   2. OPTIMIZE prompts using GEPA/MIPROv2 with training data         │
    │   3. EXPORT the optimized prompts                                    │
    │   4. USE in your production framework (LangChain, CrewAI, etc.)     │
    │                                                                      │
    │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
    │   │   DSPy       │     │   GEPA       │     │  Optimized   │        │
    │   │  Signature   │ ──► │  Optimizer   │ ──► │   Prompts    │        │
    │   │  + Examples  │     │  + Feedback  │     │              │        │
    │   └──────────────┘     └──────────────┘     └──────┬───────┘        │
    │                                                     │                │
    │                                                     ▼                │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │                    TARGET FRAMEWORKS                         │   │
    │   ├──────────┬──────────┬──────────┬──────────┬────────────────┤   │
    │   │ LangChain│  CrewAI  │  AutoGen │ Pydantic │ OpenAI Agents  │   │
    │   │          │          │          │    AI    │                │   │
    │   └──────────┴──────────┴──────────┴──────────┴────────────────┘   │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

    WORKFLOW EXAMPLE:

    Step 1: Define task in DSPy
    ─────────────────────────────
    class WeatherAnalysis(dspy.Signature):
        \"\"\"Analyze weather data for insurance CAT event determination.\"\"\"
        weather_data: str = dspy.InputField()
        cat_status: str = dspy.OutputField()
        reasoning: str = dspy.OutputField()

    Step 2: Optimize with GEPA
    ─────────────────────────────
    optimizer = dspy.GEPA(metric=accuracy_metric, max_iterations=5)
    optimized = optimizer.compile(
        dspy.ChainOfThought(WeatherAnalysis),
        trainset=labeled_examples,
        feedback_fn=domain_expert_feedback
    )

    Step 3: Extract optimized prompts
    ─────────────────────────────────
    # DSPy stores optimized instructions in the module
    optimized_prompt = optimized.extended_signature.instructions
    # Or inspect the full prompt template
    print(optimized.dump_state())

    Step 4: Use in target framework
    ─────────────────────────────────

    # LangChain
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI()
    response = llm.invoke(optimized_prompt + "\\n\\nWeather: {data}")

    # CrewAI
    from crewai import Agent
    agent = Agent(
        role="Weather Analyst",
        goal="Analyze weather for CAT events",
        backstory=optimized_prompt,  # Use optimized prompt as backstory
        ...
    )

    # AutoGen
    from autogen_agentchat.agents import AssistantAgent
    agent = AssistantAgent(
        name="WeatherAnalyst",
        system_message=optimized_prompt,  # Use optimized prompt
        ...
    )

    ═══════════════════════════════════════════════════════════════════════

    VALUE OF DSPY FOR PROMPT PORTING TO NEW MODELS:

    When you switch from GPT-4 to Claude or Llama:

    1. Your hand-crafted prompts may not transfer well
    2. DSPy can RE-OPTIMIZE for the new model automatically
    3. Just change the LM configuration and run GEPA again

    ┌─────────────────┐                    ┌─────────────────┐
    │  GPT-4 Prompt   │   Model Change     │  Claude Prompt  │
    │  (optimized)    │ ───────────────►   │  (re-optimized) │
    │                 │   + GEPA rerun     │                 │
    └─────────────────┘                    └─────────────────┘

    # Example: Porting prompts to new model

    # Original optimization for GPT-4
    dspy.configure(lm=dspy.LM("openai/gpt-4o"))
    gpt4_optimized = gepa.compile(module, trainset=data)

    # Re-optimize for Claude (same trainset, different model)
    dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-20250514"))
    claude_optimized = gepa.compile(module, trainset=data)

    # Re-optimize for Llama (same trainset, different model)
    dspy.configure(lm=dspy.LM("ollama/llama3.2"))
    llama_optimized = gepa.compile(module, trainset=data)

    Each model gets prompts tailored to its specific characteristics!
    """)


if __name__ == "__main__":
    # Run the main demo
    result = run_dspy_demo()

    # Show optimization capabilities
    if result:
        run_dspy_optimization_demo()

    # Show cross-framework usage
    show_cross_framework_optimization()

    # Optional: MLFlow demo (requires mlflow installed)
    try:
        import mlflow
        run_mlflow_integration_demo()
    except ImportError:
        print("\n(Skipping MLFlow demo - install with: pip install mlflow)")
