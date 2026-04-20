import json
import re


def score_rule_based(output: str, query: str) -> tuple[float, str]:
    """
    Score output using deterministic rules.
    Returns (score 0.0-1.0, feedback string).
    """
    if not output or not output.strip():
        return 0.0, "Output is empty."

    scores = []
    feedback_parts = []

    # Rule 1: Minimum length (at least 20 chars)
    if len(output.strip()) >= 20:
        scores.append(1.0)
    else:
        scores.append(0.0)
        feedback_parts.append("Output is too short (< 20 chars).")

    # Rule 2: Not just repeating the query verbatim
    if output.strip().lower() != query.strip().lower():
        scores.append(1.0)
    else:
        scores.append(0.0)
        feedback_parts.append("Output merely repeats the query.")

    # Rule 3: No placeholder text
    placeholders = ["todo", "placeholder", "n/a", "none", "unknown", "..."]
    has_placeholder = any(p in output.lower() for p in placeholders)
    if not has_placeholder:
        scores.append(1.0)
    else:
        scores.append(0.3)
        feedback_parts.append("Output contains placeholder text.")

    # Rule 4: Responds to at least one keyword from the query
    query_keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))
    output_lower = output.lower()
    keyword_hits = sum(1 for kw in query_keywords if kw in output_lower)
    keyword_score = min(keyword_hits / max(len(query_keywords), 1), 1.0)
    scores.append(keyword_score)
    if keyword_score < 0.3:
        feedback_parts.append("Output does not address the query keywords.")

    # Rule 5: No obvious toxic phrases
    toxic_patterns = [r'\bhate\b', r'\bkill\b', r'\bstupid\b', r'\bidiot\b']
    is_toxic = any(re.search(p, output.lower()) for p in toxic_patterns)
    if not is_toxic:
        scores.append(1.0)
    else:
        scores.append(0.0)
        feedback_parts.append("Output contains potentially harmful language.")

    final_score = sum(scores) / len(scores)
    feedback = " ".join(feedback_parts) if feedback_parts else "Output passed all rule-based checks."
    return round(final_score, 3), feedback


def validate_json_schema(output: str, schema: dict) -> tuple[bool, str]:
    """Check that output is valid JSON matching a simple required-keys schema."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        return False, f"Missing required keys: {missing}"

    return True, "JSON schema valid."
