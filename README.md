# LangValidator — Self-Validating Agent Graph

A production-style LangGraph pipeline where every agent output passes through a **checkpoint node** before proceeding. The checkpoint scores the output, then routes to the next agent, retries with feedback, or halts with a reason.

---

## How it works

```
retrieval_agent
      │
      ▼
retrieval_checkpoint ──── score < 0.45 or retries exhausted ──► halt
      │                                    ▲
      │ score ≥ 0.75              retry ───┘
      ▼
reasoning_agent
      │
      ▼
reasoning_checkpoint ──── score < 0.45 or retries exhausted ──► halt
      │                                    ▲
      │ score ≥ 0.75              retry ───┘
      ▼
   finalize
```

- **Pass** (score ≥ 0.75) → move to the next stage
- **Retry** (0.45 ≤ score < 0.75, retries remaining) → loop back with feedback injected into the prompt
- **Halt** (score < 0.45, or retries exhausted) → stop with a reason

---

## Project structure

```
LangValidator/
├── graph/
│   ├── state.py            # AgentState TypedDict
│   ├── nodes.py            # retrieval_agent, reasoning_agent, checkpoint_node, halt_node, finalize_node
│   └── graph_builder.py    # StateGraph wiring, conditional edges, retry counters
├── validators/
│   ├── rule_based.py       # Deterministic scoring (length, keywords, toxicity, placeholders)
│   ├── llm_judge.py        # Claude Haiku as LLM-as-judge (0–10 rubric → 0.0–1.0)
│   └── semantic.py         # Cosine similarity helper for embedding-based scoring
├── tools/                  # MCP-style composable tools
│   ├── fact_check.py       # Checks output against a knowledge base
│   └── schema_check.py     # Validates output as JSON matching a required-keys schema
├── examples/
│   ├── run_rule_based.py       # Full pipeline with rule-based scorer
│   ├── run_llm_judge.py        # Full pipeline with LLM-as-judge scorer
│   └── standalone_validators.py # Validators and tools without running the graph
├── tests/
│   └── test_validators.py  # 18 unit tests — no API key required
├── main.py                 # CLI entry point
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the pipeline

```bash
# Default query, LLM-as-judge scoring
python main.py

# Custom query with rule-based scoring
python main.py --query "Explain what LangGraph is" --strategy rule_based

# More retries
python main.py --query "What is Anthropic?" --max-retries 3
```

### 4. Run examples

```bash
# No API key needed — pure Python logic
python examples/standalone_validators.py

# Full pipeline — rule-based scorer
python examples/run_rule_based.py

# Full pipeline — LLM-as-judge scorer
python examples/run_llm_judge.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

All 18 tests run without an API key.

---

## Scoring strategies

| Strategy | How it works | When to use |
|---|---|---|
| `rule_based` | Checks length, keyword overlap, placeholders, toxicity | Fast, deterministic, no extra LLM calls |
| `llm_judge` | Claude Haiku rates output 0–10 using a rubric | More nuanced, catches subtle quality issues |

Both strategies are blended with a **fact-check tool** (30% weight) that verifies the output against a knowledge base.

Switch strategies via the `--strategy` flag or the `SCORING_STRATEGY` environment variable.

---

## Validators and tools

### `validators/rule_based.py`
Scores output deterministically across five rules:
- Minimum length (≥ 20 chars)
- Does not merely repeat the query
- No placeholder text (`TODO`, `N/A`, `...`)
- Keyword overlap with the query
- No toxic language patterns

### `validators/llm_judge.py`
Sends the query + output to Claude Haiku with a 0–10 rubric. Returns a normalized score and one-sentence reasoning.

### `validators/semantic.py`
Pure-Python cosine similarity. Plug in any embedding function (`embed_fn`) to score semantic relevance between query and output.

### `tools/fact_check.py`
MCP-style tool. Checks whether statements in the output align with a knowledge base. Returns a confidence score and issue list.

### `tools/schema_check.py`
MCP-style tool. Validates that output is valid JSON with required keys, correct types, and non-empty fields.

---

## State schema

```python
class AgentState(TypedDict):
    query: str
    conversation: List[dict]          # full message history
    current_agent: str                # which agent just ran
    current_agent_output: Optional[str]
    validation_score: Optional[float] # 0.0 – 1.0
    validation_feedback: Optional[str]
    retry_count: int
    max_retries: int
    halted: bool
    halt_reason: Optional[str]
    final_answer: Optional[str]
```

---

## What you learn by building this

| Concept | Where it appears |
|---|---|
| LangGraph nodes and edges | `graph/graph_builder.py` |
| Conditional routing | `route_after_retrieval_checkpoint`, `route_after_reasoning_checkpoint` |
| Cycle / retry loops | `retry_*_counter` nodes looping back |
| State persistence across nodes | `AgentState` passed through every node |
| LLM-as-judge pattern | `validators/llm_judge.py` |
| MCP-style composable tools | `tools/fact_check.py`, `tools/schema_check.py` |
| Agentic self-correction | Retry node injects `validation_feedback` into the next prompt |

---

## Requirements

- Python 3.11+
- `anthropic >= 0.40.0`
- `langgraph >= 0.2.0`
- `langchain >= 0.3.0`
- `pytest >= 8.0.0`
