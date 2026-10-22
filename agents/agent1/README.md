# Agent 1

Agent 1 is a self-contained LangGraph skeleton for code-quality and dataset
analysis. The current nodes are intentionally deterministic stubs. They print
their node name and return placeholder state updates so the graph contract can
be integrated before real analysis is added.

## Run locally

From the repository root, with the project virtualenv active:

```bash
python -m agents.agent1.run_local
```

The runner prints the graph as Mermaid and then prints the dummy output JSON.
The graph can also be rendered directly:

```python
from agents.agent1 import build_graph

app = build_graph()
print(app.get_graph().draw_mermaid())
```

## Integration

The public package API exports exactly:


Agent 1 owns `Agent1State` in `state.py`. Parent graph integration should use
`adapter.py` and change only `from_main_state()` and `to_main_state()` when the
parent state contract is available. No import performs environment reads,
printing, or file creation.

Runtime workspace copies and snapshots belong in a temporary directory or
`.workspace/`; the latter is ignored by this package.

## Output contract

```json
{
  "code_quality": {
    "issues_before": 0,
    "issues_after": 0,
    "changes": [],
    "verified": true,
    "baseline_ok": true
  },
  "problem_type": "classification",
  "subtype": "binary",
  "target": "target",
  "features": {"numeric": [], "categorical": []},
  "confidence": 0.0
}
```

## LLM edits

The plan is deterministic: it is built from missing-docstring and missing-type-
hint quality findings. The LLM is used only to edit one function at a time.

Use Groq with the `openai/gpt-oss-20b` model, or set
`AGENT1_LLM_BASE_URL`, `AGENT1_LLM_MODEL`, `AGENT1_LLM_TIMEOUT`, and
`AGENT1_ENABLE_LLM`. Defaults are base URL `https://api.groq.com/openai/v1`, model
`openai/gpt-oss-20b`, API key from `AGENT1_LLM_API_KEY` or `GROQ_API_KEY`,
timeout 120 seconds, and at most 10 items.

The exact prompt is:

```text
System: You are a careful Python code improver. You receive one function and an issue. Return ONLY the improved function as plain Python code, with no explanation and no markdown. Never change the behavior or logic of the function.

User: Issue: {issue}

Function:
{function_source}
```

Answers are rejected when they do not parse as the same function, change
parameters or logic, omit the requested docstring/type hints, lose comments, or
exceed three times the original length. Markdown fences are stripped before
validation. An unavailable Groq endpoint skips LLM edits without failing the graph.

## Local environment file

Copy `.env.example` to `.env` inside `agents/agent1/` and set `GROQ_API_KEY`
(or `AGENT1_LLM_API_KEY`) and optionally `AGENT1_LLM_MODEL` or
`AGENT1_LLM_BASE_URL`. The file is loaded lazily from the package location;
variables already present in the process environment always take precedence.
The ML client uses `ML_LLM_MAX_TOKENS` (default `4000`) and optionally sends
`ML_LLM_REASONING_EFFORT` when set to `low`, `medium`, or `high`.
The real `.env` is ignored by Git and secret values are never included in Agent
1 output or logs.
