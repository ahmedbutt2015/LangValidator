"""
MCP-style schema validation tool.
Validates that agent outputs conform to expected JSON structures.
"""

import json


def check_json_schema(output: str, schema: dict) -> tuple[float, str]:
    """
    Validate output against a simple schema descriptor.

    schema format:
      {
        "required": ["key1", "key2"],
        "types": {"key1": "str", "key2": "int"},   # optional
        "non_empty": ["key1"]                        # optional: these values must be non-empty
      }

    Returns (score 0.0-1.0, feedback).
    """
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return 0.0, "Output is not valid JSON."

    if not isinstance(data, dict):
        return 0.2, "Output JSON is not an object/dict."

    required = schema.get("required", [])
    type_checks = schema.get("types", {})
    non_empty = schema.get("non_empty", [])

    issues = []
    total_checks = len(required) + len(type_checks) + len(non_empty)
    if total_checks == 0:
        return 1.0, "No schema constraints defined — assumed OK."

    passed = 0

    for key in required:
        if key in data:
            passed += 1
        else:
            issues.append(f"Missing required key: '{key}'.")

    type_map = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}
    for key, expected_type_name in type_checks.items():
        expected_type = type_map.get(expected_type_name)
        if expected_type and key in data:
            if isinstance(data[key], expected_type):
                passed += 1
            else:
                issues.append(f"Key '{key}' expected {expected_type_name}, got {type(data[key]).__name__}.")
        else:
            passed += 1  # can't check if key is missing or type unknown

    for key in non_empty:
        if key in data and data[key]:
            passed += 1
        elif key in data:
            issues.append(f"Key '{key}' is present but empty.")
        # if key missing, already caught by required check

    score = round(passed / total_checks, 2)
    feedback = " ".join(issues) if issues else "Output satisfies the JSON schema."
    return score, feedback
