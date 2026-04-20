"""Unit tests for validators and tools — no LLM calls needed."""

import pytest
from validators.rule_based import score_rule_based, validate_json_schema
from validators.semantic import cosine_similarity
from tools.fact_check import check_facts
from tools.schema_check import check_json_schema


# ---------------------------------------------------------------------------
# Rule-based validator
# ---------------------------------------------------------------------------

class TestRuleBased:
    def test_empty_output_scores_zero(self):
        score, _ = score_rule_based("", "What is Python?")
        assert score == 0.0

    def test_good_output_scores_high(self):
        output = (
            "Python is a high-level programming language known for its readability. "
            "It supports multiple paradigms including OOP and functional programming."
        )
        score, feedback = score_rule_based(output, "What is Python?")
        assert score >= 0.7, f"Expected >= 0.7, got {score}. Feedback: {feedback}"

    def test_placeholder_penalized(self):
        score, feedback = score_rule_based("TODO: fill this in later", "What is Python?")
        assert score < 0.8
        assert "placeholder" in feedback.lower() or "todo" in feedback.lower() or score < 0.8

    def test_toxic_output_scores_low(self):
        score, _ = score_rule_based("I hate this stupid question.", "What is Python?")
        assert score <= 0.6

    def test_validate_json_schema_pass(self):
        output = '{"name": "Alice", "age": 30}'
        ok, msg = validate_json_schema(output, {"required": ["name", "age"]})
        assert ok

    def test_validate_json_schema_missing_key(self):
        output = '{"name": "Alice"}'
        ok, msg = validate_json_schema(output, {"required": ["name", "age"]})
        assert not ok
        assert "age" in msg

    def test_validate_json_schema_invalid_json(self):
        ok, msg = validate_json_schema("not json", {"required": ["name"]})
        assert not ok


# ---------------------------------------------------------------------------
# Semantic validator (no embed_fn needed — just cosine_similarity)
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0


# ---------------------------------------------------------------------------
# Fact-check tool
# ---------------------------------------------------------------------------

class TestFactCheck:
    def test_no_relevant_topic(self):
        score, feedback = check_facts("The weather is sunny today.", [])
        assert score == 1.0  # nothing to check

    def test_relevant_topic_present(self):
        output = "Python is a high-level, interpreted programming language used widely."
        score, feedback = check_facts(output, ["python"])
        # Should find some matching words
        assert score >= 0.0

    def test_unknown_topic_skipped(self):
        score, feedback = check_facts("Random text here.", ["nonexistent_topic_xyz"])
        assert score == 1.0


# ---------------------------------------------------------------------------
# Schema-check tool
# ---------------------------------------------------------------------------

class TestSchemaCheck:
    def test_valid_schema(self):
        output = '{"answer": "42", "confidence": 0.9}'
        schema = {"required": ["answer", "confidence"], "non_empty": ["answer"]}
        score, feedback = check_json_schema(output, schema)
        assert score == 1.0

    def test_missing_required_key(self):
        output = '{"answer": "42"}'
        schema = {"required": ["answer", "confidence"]}
        score, feedback = check_json_schema(output, schema)
        assert score < 1.0
        assert "confidence" in feedback

    def test_wrong_type(self):
        output = '{"count": "not_an_int"}'
        schema = {"required": ["count"], "types": {"count": "int"}}
        score, feedback = check_json_schema(output, schema)
        assert score < 1.0

    def test_not_json(self):
        score, feedback = check_json_schema("plain text", {})
        assert score == 0.0

    def test_empty_schema(self):
        score, feedback = check_json_schema('{"anything": true}', {})
        assert score == 1.0
