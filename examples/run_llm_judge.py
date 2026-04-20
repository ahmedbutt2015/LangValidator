"""
Example: run the pipeline with LLM-as-judge scoring (Claude Haiku rates each output).

Run:
  ANTHROPIC_API_KEY=sk-... python examples/run_llm_judge.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run

if __name__ == "__main__":
    run(
        query="Explain the difference between LangGraph and a simple LangChain chain.",
        max_retries=2,
        strategy="llm_judge",
    )
