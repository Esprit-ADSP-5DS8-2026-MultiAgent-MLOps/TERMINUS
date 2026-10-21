"""Tests for deterministic quality analysis."""

from pathlib import Path

from agents.agent1.nodes.quality_analysis import quality_analysis


def test_quality_analysis_reports_custom_issues(tmp_path: Path) -> None:
    """AST checks report missing documentation, annotations, and absolute paths."""
    (tmp_path / "module.py").write_text(
        "def compute(value):\n    return '/tmp/output'\n", encoding="utf-8"
    )
    result = quality_analysis({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    issues = result["quality_issues"]
    codes = {issue["code"] for issue in issues}  # type: ignore[union-attr]
    assert {"DOC001", "ANN001", "PATH001"} <= codes
    assert result["quality_before"]["total"] == len(issues)  # type: ignore[index]


def test_quality_analysis_handles_missing_workspace(tmp_path: Path) -> None:
    """A missing workspace becomes a controlled error state."""
    result = quality_analysis({"workspace_path": str(tmp_path / "missing")})  # type: ignore[arg-type]
    assert result["quality_issues"] == []
    assert result["errors"]  # type: ignore[index]
