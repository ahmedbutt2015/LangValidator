"""
Example: use the validators and tools standalone, without running the graph.
Useful for understanding the scoring logic in isolation.

Run:
  python examples/standalone_validators.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validators.rule_based import score_rule_based, validate_json_schema
from validators.semantic import cosine_similarity
from tools.fact_check import check_facts
from tools.schema_check import check_json_schema


def demo_rule_based():
    print("=== Rule-Based Validator ===")
    cases = [
        ("Python is a high-level interpreted language popular for data science.", "What is Python?"),
        ("TODO: placeholder", "What is Python?"),
        ("", "What is Python?"),
        ("I hate this stupid language.", "What is Python?"),
    ]
    for output, query in cases:
        score, feedback = score_rule_based(output, query)
        label = repr(output[:40])
        print(f"  Output: {label}")
        print(f"  Score:  {score}  |  Feedback: {feedback}\n")


def demo_json_schema():
    print("=== JSON Schema Validator ===")
    schema = {"required": ["answer", "confidence"], "types": {"confidence": "float"}, "non_empty": ["answer"]}

    valid_output = '{"answer": "LangChain is a framework for LLM apps.", "confidence": 0.92}'
    invalid_output = '{"answer": "", "confidence": "high"}'

    for output in [valid_output, invalid_output]:
        score, feedback = check_json_schema(output, schema)
        print(f"  Output: {output[:60]}")
        print(f"  Score:  {score}  |  Feedback: {feedback}\n")


def demo_fact_check():
    print("=== Fact-Check Tool ===")
    outputs = [
        "Python is a high-level, interpreted programming language.",
        "Python is a low-level compiled systems language like C.",
        "The weather today is sunny and warm.",
    ]
    for output in outputs:
        score, feedback = check_facts(output, ["python"])
        print(f"  Output: {output[:60]}")
        print(f"  Score:  {score}  |  Feedback: {feedback}\n")


def demo_cosine():
    print("=== Cosine Similarity (semantic building block) ===")
    pairs = [
        ([1.0, 0.0], [1.0, 0.0], "identical"),
        ([1.0, 0.0], [0.0, 1.0], "orthogonal"),
        ([1.0, 1.0], [1.0, 0.5], "similar"),
    ]
    for a, b, label in pairs:
        sim = cosine_similarity(a, b)
        print(f"  {label}: cosine({a}, {b}) = {sim:.3f}")
    print()


if __name__ == "__main__":
    demo_rule_based()
    demo_json_schema()
    demo_fact_check()
    demo_cosine()
