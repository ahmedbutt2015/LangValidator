"""
Entry point for the Self-Validating Agent Graph.

Usage:
  python main.py
  python main.py --query "Explain what LangGraph is"
  python main.py --strategy rule_based --max-retries 3
"""

import argparse
import os
import sys

from graph.graph_builder import build_graph


def run(query: str, max_retries: int = 2, strategy: str = "llm_judge") -> None:
    os.environ["SCORING_STRATEGY"] = strategy

    initial_state = {
        "query": query,
        "conversation": [],
        "current_agent": "",
        "current_agent_output": None,
        "validation_score": None,
        "validation_feedback": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "halted": False,
        "halt_reason": None,
        "final_answer": None,
    }

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Strategy: {strategy}  |  Max retries: {max_retries}")
    print(f"{'='*60}\n")

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    if final_state.get("halted"):
        print("RESULT: HALTED")
        print(f"Reason: {final_state.get('halt_reason')}")
    else:
        print("RESULT: SUCCESS")
        print(f"Final answer:\n{final_state.get('final_answer')}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Self-Validating Agent Graph")
    parser.add_argument(
        "--query",
        type=str,
        default="What is LangGraph and how does it relate to LangChain?",
    )
    parser.add_argument(
        "--strategy",
        choices=["rule_based", "llm_judge"],
        default="llm_judge",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    run(query=args.query, max_retries=args.max_retries, strategy=args.strategy)


if __name__ == "__main__":
    main()
