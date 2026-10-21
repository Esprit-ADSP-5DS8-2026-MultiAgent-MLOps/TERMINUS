"""Run the Agent 1 skeleton locally."""

import argparse
import json
from pathlib import Path

from . import build_graph
from .runner import _run_agent1_state


def parse_args() -> argparse.Namespace:
    """Parse optional sample project and CSV paths."""
    repository_root = Path(__file__).resolve().parents[2]
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Print the graph and invoke it with sample input paths."""
    parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    project_path = repository_root / "eval" / "projects" / "credit_risk"
    csv_path = project_path / "data.csv"
    app = build_graph()
    print(app.get_graph().draw_mermaid())
    state = _run_agent1_state(str(project_path), str(csv_path))
    baseline = state.get("baseline") or {}
    print(
        json.dumps(
            {
                "baseline": {
                    "entry_point": baseline.get("entry_point"),
                    "exit_code": baseline.get("exit_code"),
                    "stderr": baseline.get("stderr", ""),
                },
                "allowed_tiers": state.get("allowed_tiers", []),
                "changes": state.get("changes", []),
            },
            indent=2,
        )
    )
    print(json.dumps(state["output"], indent=2))


if __name__ == "__main__":
    main()
