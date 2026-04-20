import math


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    denom = _norm(vec_a) * _norm(vec_b)
    if denom == 0:
        return 0.0
    return _dot(vec_a, vec_b) / denom


def score_semantic(
    output: str,
    query: str,
    embed_fn,  # callable(text: str) -> list[float]
) -> tuple[float, str]:
    """
    Embed query and output, return cosine similarity as the score.
    embed_fn must be provided by the caller (e.g. OpenAI embeddings, sentence-transformers).
    """
    if not output or not output.strip():
        return 0.0, "Output is empty — cannot compute semantic similarity."

    vec_query = embed_fn(query)
    vec_output = embed_fn(output)
    sim = cosine_similarity(vec_query, vec_output)
    score = round(max(0.0, min(sim, 1.0)), 3)

    if score >= 0.75:
        feedback = f"Semantic similarity {score:.2f} — output is highly relevant to the query."
    elif score >= 0.45:
        feedback = f"Semantic similarity {score:.2f} — output is somewhat relevant; consider refining."
    else:
        feedback = f"Semantic similarity {score:.2f} — output appears off-topic."

    return score, feedback
