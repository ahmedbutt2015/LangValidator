"""
Builds and compiles the LangGraph validation workflow.

Flow:
  retrieval_agent → checkpoint → [pass → reasoning_agent → checkpoint → [pass → finalize]
                                                                        [retry → reasoning_agent]
                                                                        [halt → halt_node]]
                                 [retry → retrieval_agent]
                                 [halt → halt_node]
"""

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    retrieval_agent,
    reasoning_agent,
    checkpoint_node,
    halt_node,
    finalize_node,
)

PASS_THRESHOLD = 0.75
RETRY_THRESHOLD = 0.45


def route_after_retrieval_checkpoint(state: AgentState) -> str:
    score = state.get("validation_score", 0.0)
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if score >= PASS_THRESHOLD:
        return "reasoning_agent"
    elif score >= RETRY_THRESHOLD and retries < max_retries:
        return "retry_retrieval"
    else:
        return "halt"


def route_after_reasoning_checkpoint(state: AgentState) -> str:
    score = state.get("validation_score", 0.0)
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if score >= PASS_THRESHOLD:
        return "finalize"
    elif score >= RETRY_THRESHOLD and retries < max_retries:
        return "retry_reasoning"
    else:
        return "halt"


def increment_retry(state: AgentState) -> dict:
    """Thin node that increments the retry counter before looping back."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("retrieval_checkpoint", checkpoint_node)
    graph.add_node("retry_retrieval_counter", increment_retry)

    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("reasoning_checkpoint", checkpoint_node)
    graph.add_node("retry_reasoning_counter", increment_retry)

    graph.add_node("halt_node", halt_node)
    graph.add_node("finalize_node", finalize_node)

    # --- Entry point ---
    graph.set_entry_point("retrieval_agent")

    # --- Edges from retrieval ---
    graph.add_edge("retrieval_agent", "retrieval_checkpoint")
    graph.add_conditional_edges(
        "retrieval_checkpoint",
        route_after_retrieval_checkpoint,
        {
            "reasoning_agent": "reasoning_agent",
            "retry_retrieval": "retry_retrieval_counter",
            "halt": "halt_node",
        },
    )
    graph.add_edge("retry_retrieval_counter", "retrieval_agent")

    # --- Edges from reasoning ---
    graph.add_edge("reasoning_agent", "reasoning_checkpoint")
    graph.add_conditional_edges(
        "reasoning_checkpoint",
        route_after_reasoning_checkpoint,
        {
            "finalize": "finalize_node",
            "retry_reasoning": "retry_reasoning_counter",
            "halt": "halt_node",
        },
    )
    graph.add_edge("retry_reasoning_counter", "reasoning_agent")

    # --- Terminal nodes ---
    graph.add_edge("halt_node", END)
    graph.add_edge("finalize_node", END)

    return graph.compile()
