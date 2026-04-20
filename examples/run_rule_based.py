"""
Example: run the pipeline with the rule-based scorer (no LLM judge calls for scoring).
Still uses Claude for the agent nodes themselves.

Run:
  ANTHROPIC_API_KEY=sk-... python examples/run_rule_based.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run

if __name__ == "__main__":
    run(
        query="What is LangChain and what problems does it solve?",
        max_retries=2,
        strategy="rule_based",
    )
