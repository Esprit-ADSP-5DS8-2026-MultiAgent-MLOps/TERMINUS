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

- `build_graph()`: returns the compiled LangGraph application.
- `run_agent1(project_path, csv_path)`: invokes the graph and returns output JSON.

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

The real implementation will preserve this shape while replacing the dummy
node behavior with the safety and analysis steps described by the team
contract.
