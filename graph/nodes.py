"""
Agent nodes and the checkpoint node for the LangGraph validation pipeline.
"""

import os
from anthropic import Anthropic
from graph.state import AgentState
from validators.rule_based import score_rule_based
from validators.llm_judge import score_llm_judge
from tools.fact_check import check_facts

client = Anthropic()

# ---------------------------------------------------------------------------
# Scoring strategy — change this to switch between rule_based / llm_judge
# ---------------------------------------------------------------------------
SCORING_STRATEGY = os.getenv("SCORING_STRATEGY", "llm_judge")  # "rule_based" | "llm_judge"

PASS_THRESHOLD = 0.75
RETRY_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def retrieval_agent(state: AgentState) -> dict:
    """Simulates a retrieval agent that fetches relevant context."""
    query = state["query"]
    conversation = state.get("conversation", [])

    messages = conversation + [
        {
            "role": "user",
            "content": (
                f"You are a retrieval agent. Given the query below, summarize the most relevant "
                f"factual information you know about it. Be concise (2-4 sentences).\n\nQuery: {query}"
            ),
        }
    ]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=messages,
    )
    output = response.content[0].text.strip()

    return {
        "current_agent": "retrieval_agent",
        "current_agent_output": output,
        "conversation": conversation + [
            {"role": "user", "content": f"Query: {query}"},
            {"role": "assistant", "content": output},
        ],
    }


def reasoning_agent(state: AgentState) -> dict:
    """Takes retrieved context and produces a full reasoned answer."""
    query = state["query"]
    conversation = state.get("conversation", [])
    prior_output = state.get("current_agent_output", "")
    feedback = state.get("validation_feedback", "")

    retry_hint = (
        f"\n\nNote from validator (score too low): {feedback}\nPlease improve your answer."
        if feedback and state.get("retry_count", 0) > 0
        else ""
    )

    messages = conversation + [
        {
            "role": "user",
            "content": (
                f"You are a reasoning agent. Using the context below, provide a thorough and accurate "
                f"answer to the query.{retry_hint}\n\nContext:\n{prior_output}\n\nQuery: {query}"
            ),
        }
    ]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=messages,
    )
    output = response.content[0].text.strip()

    return {
        "current_agent": "reasoning_agent",
        "current_agent_output": output,
        "conversation": conversation + [
            {"role": "assistant", "content": output},
        ],
    }


# ---------------------------------------------------------------------------
# Checkpoint node
# ---------------------------------------------------------------------------

def checkpoint_node(state: AgentState) -> dict:
    """
    Scores the current agent output and updates the state with:
      - validation_score
      - validation_feedback
    Does NOT make routing decisions — that is handled by route_after_checkpoint.
    """
    output = state.get("current_agent_output", "")
    query = state["query"]

    if SCORING_STRATEGY == "rule_based":
        score, feedback = score_rule_based(output, query)
    else:
        # LLM judge (default)
        score, feedback = score_llm_judge(output, query)

    # Always also run the fact-checker as an MCP-style tool
    fact_score, fact_feedback = check_facts(output)
    # Blend: 70% primary scorer, 30% fact check
    blended_score = round(0.7 * score + 0.3 * fact_score, 3)
    combined_feedback = f"[Primary] {feedback} [FactCheck] {fact_feedback}"

    print(
        f"\n--- Checkpoint ---\n"
        f"Agent: {state.get('current_agent')}\n"
        f"Primary score ({SCORING_STRATEGY}): {score}\n"
        f"Fact-check score: {fact_score}\n"
        f"Blended score: {blended_score}\n"
        f"Feedback: {combined_feedback}\n"
        f"Retry count: {state.get('retry_count', 0)}\n"
        f"------------------"
    )

    return {
        "validation_score": blended_score,
        "validation_feedback": combined_feedback,
    }


# ---------------------------------------------------------------------------
# Halt node
# ---------------------------------------------------------------------------

def halt_node(state: AgentState) -> dict:
    reason = (
        f"Halted after {state.get('retry_count', 0)} retries. "
        f"Final score: {state.get('validation_score', 'N/A')}. "
        f"Feedback: {state.get('validation_feedback', '')}"
    )
    print(f"\n[HALT] {reason}\n")
    return {
        "halted": True,
        "halt_reason": reason,
        "final_answer": f"Could not produce a satisfactory answer. {reason}",
    }


# ---------------------------------------------------------------------------
# Finalize node
# ---------------------------------------------------------------------------

def finalize_node(state: AgentState) -> dict:
    answer = state.get("current_agent_output", "")
    print(f"\n[DONE] Final answer produced (score={state.get('validation_score')}).\n")
    return {"final_answer": answer, "halted": False}
