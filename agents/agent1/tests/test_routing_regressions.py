"""Regression tests for verification and conditional routing."""

from pathlib import Path

from agents.agent1.graph import route_after_verify
from agents.agent1.nodes.verify import verify


def test_no_changes_verify_passes_without_a_workspace() -> None:
    """An empty change list is a successful no-op verification."""
    result = verify(
        {
            "changes": [],
            "workspace_path": "/does/not/exist",
            "baseline_ok": False,
            "quality_before": {"total": 2},
            "quality_issues": [{"code": "DOC001"}],
        }  # type: ignore[arg-type]
    )
    assert result["verification_passed"] is True
    assert result["quality_after"] == {"total": 2}


def test_failed_baseline_still_verifies_tier_one_change(tmp_path: Path) -> None:
    """Tier 1 changes pass layers 1-3 when no baseline exists for layer 4."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text(
        "def main() -> None:\n    return None\n", encoding="utf-8"
    )
    (workspace / "main.py").write_text(
        '"""Added documentation."""\n\ndef main() -> None:\n    return None\n',
        encoding="utf-8",
    )
    result = verify(
        {
            "workspace_path": str(workspace),
            "snapshot_path": str(snapshot),
            "baseline_ok": False,
            "changes": [{"file": "main.py", "tier": 1}],
            "attempts": 0,
            "quality_before": {},
        }  # type: ignore[arg-type]
    )
    assert result["verification_passed"] is True


def test_empty_plan_routes_directly_to_analysis() -> None:
    """A failed verification with no plan does not enter retry."""
    assert (
        route_after_verify(
            {"verification_passed": False, "attempts": 0, "improvement_plan": []}  # type: ignore[arg-type]
        )
        == "analyze_data"
    )
