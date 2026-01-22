"""
AI Agent Framework Decision Flowchart
=====================================

Interactive decision flowchart to help choose the right AI agent framework.
Can be run as a script or imported in a Jupyter notebook.

Usage:
    python decision_flowchart.py          # Run interactive CLI wizard
    python decision_flowchart.py --graph  # Generate visual flowchart image

Requirements:
    pip install graphviz plotly pandas
"""

import sys
from typing import Dict, List, Optional, Tuple

# Framework definitions
FRAMEWORKS = {
    "pydantic_ai": {
        "name": "Pydantic AI",
        "icon": "🎯",
        "install": "pip install pydantic-ai",
        "why": "Full Pydantic validation ensures LLM outputs match your data models exactly.",
        "best_for": ["Type safety", "Regulated industries", "Data validation"],
        "file": "pydantic_ai_demo.py",
        "color": "#7c3aed"
    },
    "langgraph": {
        "name": "LangGraph",
        "icon": "📊",
        "install": "pip install langchain langchain-openai langgraph",
        "why": "Explicit state machine design makes complex workflows debuggable.",
        "best_for": ["State machines", "Debuggability", "Complex routing"],
        "file": "langgraph_demo.py",
        "color": "#2563eb"
    },
    "openai_agents": {
        "name": "OpenAI Agents SDK",
        "icon": "⚡",
        "install": "pip install openai",
        "why": "Minimal abstraction over OpenAI function calling. Simplest multi-agent.",
        "best_for": ["Simplicity", "OpenAI-only projects", "Quick setup"],
        "file": "openai_agents_demo.py",
        "color": "#10a37f"
    },
    "anthropic": {
        "name": "Anthropic Claude SDK",
        "icon": "🤖",
        "install": "pip install anthropic",
        "why": "Direct access to Claude's capabilities including tool use.",
        "best_for": ["Claude models", "Direct API control", "Tool use"],
        "file": "anthropic_demo.py",
        "color": "#d97706"
    },
    "llamaindex": {
        "name": "LlamaIndex",
        "icon": "📚",
        "install": "pip install llama-index llama-index-llms-openai",
        "why": "Built from the ground up for RAG applications.",
        "best_for": ["RAG", "Document retrieval", "Knowledge bases"],
        "file": "llamaindex_demo.py",
        "color": "#8b5cf6"
    },
    "semantic_kernel": {
        "name": "Semantic Kernel",
        "icon": "🏢",
        "install": "pip install semantic-kernel",
        "why": "Microsoft's enterprise framework with Azure AI integration.",
        "best_for": ["Azure", "Enterprise", "Microsoft ecosystem"],
        "file": "semantic_kernel_demo.py",
        "color": "#0078d4"
    },
    "crewai": {
        "name": "CrewAI",
        "icon": "👥",
        "install": "pip install crewai crewai-tools",
        "why": "Intuitive role-based design for team workflows.",
        "best_for": ["Role-based design", "Task dependencies", "Team simulation"],
        "file": "crewai_demo.py",
        "color": "#f59e0b"
    },
    "autogen": {
        "name": "AutoGen",
        "icon": "💬",
        "install": "pip install autogen-agentchat autogen-ext[openai]",
        "why": "Powerful group chat patterns for collaborative agents.",
        "best_for": ["Group chat", "Shared context", "Collaborative agents"],
        "file": "autogen_demo.py",
        "color": "#ef4444"
    },
    "smolagents": {
        "name": "Smolagents",
        "icon": "🚀",
        "install": "pip install smolagents",
        "why": "Hugging Face's lightweight framework for rapid prototyping.",
        "best_for": ["Fast prototyping", "Minimal code", "Quick iteration"],
        "file": "smolagents_demo.py",
        "color": "#fbbf24"
    },
    "haystack": {
        "name": "Haystack",
        "icon": "🔧",
        "install": "pip install haystack-ai",
        "why": "Modular pipeline architecture for NLP applications.",
        "best_for": ["NLP pipelines", "Document processing", "Production ready"],
        "file": "haystack_demo.py",
        "color": "#22c55e"
    },
    "dspy": {
        "name": "DSPy",
        "icon": "🧬",
        "install": "pip install dspy",
        "why": "Prompt optimizer - use alongside any framework.",
        "best_for": ["Prompt optimization", "Model portability", "Systematic improvement"],
        "file": "dspy_demo.py",
        "color": "#ec4899"
    }
}

# Decision tree structure
QUESTIONS = {
    "start": {
        "text": "What's your team's experience level with AI agents?",
        "options": [
            ("New to agents (first project)", "new_team"),
            ("Some experience (evaluating options)", "experienced"),
        ]
    },
    "new_team": {
        "text": "What kind of data will your agents handle?",
        "options": [
            ("Structured/Regulated (insurance, finance, healthcare)", "pydantic_ai"),
            ("Unstructured/Flexible (general purpose)", "prototype"),
        ]
    },
    "prototype": {
        "text": "How quickly do you need to build?",
        "options": [
            ("Fast prototype (quick iteration)", "smolagents"),
            ("Production ready (long-term use)", "model_choice"),
        ]
    },
    "experienced": {
        "text": "Do you need strict type safety and data validation?",
        "options": [
            ("Yes, critical for our use case", "pydantic_ai"),
            ("Nice to have, but flexibility matters more", "model_choice"),
        ]
    },
    "model_choice": {
        "text": "Which LLM provider will you primarily use?",
        "options": [
            ("Claude (Anthropic) only", "anthropic"),
            ("OpenAI only", "openai_needs"),
            ("Multiple providers / Flexible", "workflow_type"),
        ]
    },
    "openai_needs": {
        "text": "What's your primary need with OpenAI?",
        "options": [
            ("Simple agent handoffs", "openai_agents"),
            ("Complex workflows with state", "workflow_type"),
        ]
    },
    "workflow_type": {
        "text": "What workflow pattern fits your use case?",
        "options": [
            ("Document retrieval (RAG)", "llamaindex"),
            ("Explicit state machine", "langgraph"),
            ("Role-based team design", "crewai"),
            ("Collaborative shared context", "autogen"),
            ("NLP/Document pipelines", "haystack"),
            ("Azure/Microsoft enterprise", "semantic_kernel"),
        ]
    }
}


def run_interactive_wizard():
    """Run the interactive CLI wizard."""
    print("\n" + "=" * 60)
    print("🤖 AI Agent Framework Decision Wizard")
    print("=" * 60)
    print("\nAnswer a few questions to find the best framework for your project.\n")

    current = "start"
    history = []

    while current in QUESTIONS:
        q = QUESTIONS[current]
        print(f"\n📋 {q['text']}\n")

        for i, (option_text, _) in enumerate(q["options"], 1):
            print(f"  {i}. {option_text}")

        if history:
            print(f"  0. Go back")

        while True:
            try:
                choice = input("\nYour choice: ").strip()
                if choice == "0" and history:
                    current = history.pop()
                    break
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(q["options"]):
                    history.append(current)
                    current = q["options"][choice_idx][1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

    # Show result
    if current in FRAMEWORKS:
        fw = FRAMEWORKS[current]
        print("\n" + "=" * 60)
        print(f"🎉 RECOMMENDED: {fw['icon']} {fw['name']}")
        print("=" * 60)
        print(f"\n{fw['why']}\n")
        print("Best for:", ", ".join(fw["best_for"]))
        print(f"\nInstall: {fw['install']}")
        print(f"Demo file: {fw['file']}")
        print("\n" + "=" * 60)


def create_graphviz_flowchart(output_file: str = "decision_flowchart"):
    """Generate a visual flowchart using Graphviz."""
    try:
        from graphviz import Digraph
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        print("Also install Graphviz system package: brew install graphviz (Mac) or apt install graphviz (Linux)")
        return

    dot = Digraph(comment='AI Agent Framework Decision Flowchart')
    dot.attr(rankdir='TB', size='12,16', dpi='150')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # Start node
    dot.node('start', 'START\nChoose Framework', shape='oval',
             style='filled', fillcolor='#7c3aed', fontcolor='white')

    # Question nodes
    for q_id, q_data in QUESTIONS.items():
        label = q_data['text'].replace(' ', '\n', 2)  # Wrap text
        if len(label) > 40:
            label = label[:40] + '...'
        dot.node(q_id, label, shape='diamond',
                 style='filled', fillcolor='#00d4ff', fontcolor='white')

    # Framework nodes
    for fw_id, fw_data in FRAMEWORKS.items():
        label = f"{fw_data['icon']} {fw_data['name']}\n{fw_data['best_for'][0]}"
        dot.node(fw_id, label, shape='box', style='filled,rounded',
                 fillcolor=fw_data['color'], fontcolor='white')

    # Add edges
    dot.edge('start', 'start_q', style='invis')  # Placeholder

    # Main flow
    dot.edge('start', list(QUESTIONS.keys())[0])

    for q_id, q_data in QUESTIONS.items():
        for i, (option_text, target) in enumerate(q_data['options']):
            # Truncate long labels
            edge_label = option_text[:20] + '...' if len(option_text) > 20 else option_text
            dot.edge(q_id, target, label=edge_label)

    # Render
    dot.render(output_file, format='png', cleanup=True)
    print(f"Flowchart saved to {output_file}.png")

    # Also save as SVG for web
    dot.render(output_file, format='svg', cleanup=True)
    print(f"Flowchart saved to {output_file}.svg")


def create_plotly_sankey():
    """Create an interactive Sankey diagram with Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Please install plotly: pip install plotly")
        return

    # Build node and link data
    nodes = ["Start"]
    node_colors = ["#7c3aed"]

    # Add questions
    for q_id in QUESTIONS:
        nodes.append(q_id)
        node_colors.append("#00d4ff")

    # Add frameworks
    for fw_id, fw in FRAMEWORKS.items():
        nodes.append(fw_id)
        node_colors.append(fw["color"])

    node_idx = {name: i for i, name in enumerate(nodes)}

    sources = []
    targets = []
    values = []
    labels = []

    # Start -> first question
    first_q = list(QUESTIONS.keys())[0]
    sources.append(node_idx["Start"])
    targets.append(node_idx[first_q])
    values.append(10)
    labels.append("")

    # Questions -> targets
    for q_id, q_data in QUESTIONS.items():
        for option_text, target in q_data["options"]:
            if target in node_idx:
                sources.append(node_idx[q_id])
                targets.append(node_idx[target])
                values.append(5)
                labels.append(option_text[:30])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[n if n not in FRAMEWORKS else FRAMEWORKS[n]["name"] for n in nodes],
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels
        )
    )])

    fig.update_layout(
        title_text="AI Agent Framework Decision Flow",
        font_size=12,
        height=800
    )

    fig.write_html("decision_flowchart_sankey.html")
    print("Interactive Sankey diagram saved to decision_flowchart_sankey.html")
    fig.show()


def print_quick_reference():
    """Print a quick reference table."""
    print("\n" + "=" * 80)
    print("QUICK REFERENCE: AI Agent Framework Selection")
    print("=" * 80)

    table = [
        ("Type Safety / Validation", "Pydantic AI", "Full Pydantic models, IDE autocomplete"),
        ("New Team + Regulated Data", "Pydantic AI", "Prevents hallucinated formats"),
        ("Claude Models Only", "Anthropic SDK", "Native tool use support"),
        ("OpenAI + Simple Handoffs", "OpenAI Agents", "Minimal code, just works"),
        ("Document Retrieval (RAG)", "LlamaIndex", "Built for knowledge bases"),
        ("Debuggable State Machine", "LangGraph", "LangSmith integration"),
        ("Azure / Microsoft", "Semantic Kernel", "Enterprise ecosystem"),
        ("Role-Based Teams", "CrewAI", "Intuitive agent design"),
        ("Shared Conversation", "AutoGen", "Group chat patterns"),
        ("Fast Prototype", "Smolagents", "Minimal boilerplate"),
        ("NLP Pipelines", "Haystack", "Document processing"),
        ("Prompt Optimization", "DSPy", "Works with all frameworks"),
    ]

    print(f"\n{'Priority/Need':<30} {'Framework':<20} {'Why'}")
    print("-" * 80)
    for priority, framework, why in table:
        print(f"{priority:<30} {framework:<20} {why}")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--graph" or arg == "-g":
            create_graphviz_flowchart()
        elif arg == "--sankey" or arg == "-s":
            create_plotly_sankey()
        elif arg == "--table" or arg == "-t":
            print_quick_reference()
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            print("\nOptions:")
            print("  (no args)    Run interactive wizard")
            print("  --graph, -g  Generate Graphviz flowchart image")
            print("  --sankey, -s Create Plotly Sankey diagram")
            print("  --table, -t  Print quick reference table")
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
    else:
        run_interactive_wizard()


if __name__ == "__main__":
    main()
