"""Public runner implementation for Agent 1."""

from typing import Any

from .graph import build_graph
from .state import Agent1State


def _initial_state(project_path: str, csv_path: str) -> Agent1State:
    """Create the isolated initial state used by the public and local runners."""
    return {
        "project_path": project_path,
        "workspace_path": "",
        "snapshot_path": "",
        "csv_path": csv_path,
        "errors": [],
        "baseline": None,
        "baseline_ok": False,
        "allowed_tiers": [],
        "quality_issues": [],
        "improvement_plan": [],
        "llm_note": "",
        "llm": {},
        "rejected": [],
        "changes": [],
        "verification_passed": None,
        "verification_details": None,
        "attempts": 0,
        "quality_before": {},
        "quality_after": {},
        "profile": {},
        "problem_type": None,
        "subtype": None,
        "target": None,
        "features": {"numeric": [], "categorical": []},
        "confidence": 0.0,
        "enable_ml_fixes": True,
        "behavior_changing": False,
        "pre_ml_snapshot": "",
        "ml_review": {},
        "ml_fixes": {},
        "ml_review_after": {},
        "tool_errors": [],
        "output": {},
    }


def _run_agent1_state(project_path: str, csv_path: str) -> Agent1State:
    """Run the graph and retain diagnostics for the local runner."""
    return build_graph().invoke(_initial_state(project_path, csv_path))


def run_agent1(project_path: str, csv_path: str) -> dict[str, Any]:
    """Run the graph and return its output-shaped dictionary."""
    result = _run_agent1_state(project_path, csv_path)
    return result["output"]
