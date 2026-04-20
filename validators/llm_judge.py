import re
from anthropic import Anthropic

client = Anthropic()

JUDGE_SYSTEM = """You are a strict but fair output quality judge.
You will be given a user query and an agent's response.
Rate the response on a scale from 0 to 10 using this rubric:

  10 — Perfect: completely correct, complete, safe, directly answers the query.
  7-9 — Good: mostly correct and complete with minor gaps.
  4-6 — Partial: addresses the query but has significant gaps or inaccuracies.
  1-3 — Poor: mostly off-topic, incorrect, or very incomplete.
  0   — Unacceptable: empty, harmful, or completely irrelevant.

Reply ONLY with JSON in this exact format (no extra text):
{"score": <int 0-10>, "reasoning": "<one sentence>"}"""


def score_llm_judge(output: str, query: str) -> tuple[float, str]:
    """
    Use Claude as an LLM-as-judge to score the agent output.
    Returns (normalized score 0.0-1.0, reasoning string).
    """
    if not output or not output.strip():
        return 0.0, "Output is empty."

    user_message = f"QUERY:\n{query}\n\nAGENT RESPONSE:\n{output}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Parse the JSON reply
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        # Fallback: try to extract a number
        num = re.search(r'\b(\d+)\b', raw)
        score_raw = int(num.group(1)) if num else 5
        reasoning = raw
    else:
        import json
        data = json.loads(match.group())
        score_raw = int(data.get("score", 5))
        reasoning = data.get("reasoning", "")

    normalized = round(max(0, min(score_raw, 10)) / 10.0, 2)
    return normalized, reasoning
