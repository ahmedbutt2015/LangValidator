"""
MCP-style fact-check tool.
In production, plug in a real knowledge base or retriever.
Here we use a small in-memory knowledge base to demonstrate the interface.
"""

KNOWLEDGE_BASE = {
    "python": "Python is a high-level, interpreted programming language.",
    "langgraph": "LangGraph is a library for building stateful, multi-actor LLM applications.",
    "langchain": "LangChain is a framework for developing applications powered by LLMs.",
    "anthropic": "Anthropic is an AI safety company that created Claude.",
    "claude": "Claude is a family of AI assistants built by Anthropic.",
}


def check_facts(output: str, topic_hints: list[str] | None = None) -> tuple[float, str]:
    """
    Simple fact-checking tool: checks if output contradicts known facts.
    Returns (confidence 0.0-1.0, explanation).

    topic_hints: optional list of keywords to focus the check.
    """
    output_lower = output.lower()

    checked = 0
    passed = 0
    issues = []

    topics = topic_hints or list(KNOWLEDGE_BASE.keys())

    for topic in topics:
        if topic not in KNOWLEDGE_BASE:
            continue
        if topic not in output_lower:
            continue  # output doesn't mention this topic — skip

        checked += 1
        fact = KNOWLEDGE_BASE[topic].lower()
        # Naive check: at least one key phrase from the fact appears in the output
        fact_words = [w for w in fact.split() if len(w) > 4]
        hits = sum(1 for w in fact_words if w in output_lower)
        if hits >= max(1, len(fact_words) // 3):
            passed += 1
        else:
            issues.append(f"Possible inaccuracy about '{topic}'.")

    if checked == 0:
        return 1.0, "No checkable facts found in output — assumed OK."

    score = round(passed / checked, 2)
    feedback = " ".join(issues) if issues else f"All {checked} fact(s) checked passed."
    return score, feedback
