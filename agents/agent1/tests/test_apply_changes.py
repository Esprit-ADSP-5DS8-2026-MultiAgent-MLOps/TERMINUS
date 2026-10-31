"""Tests for Tier 1 change application."""

from pathlib import Path

from agents.agent1.nodes.apply_changes import apply_changes


def test_apply_changes_is_scoped_to_workspace(tmp_path: Path) -> None:
    """The node returns only Tier 1 changes and never raises on missing tools."""
    source = tmp_path / "module.py"
    source.write_text("import os\n\nprint('ok')\n", encoding="utf-8")
    result = apply_changes({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert all(change["tier"] == 1 for change in result["changes"])
    assert all(change["file"] == "module.py" for change in result["changes"])
    assert source.exists()


def test_apply_changes_records_invalid_workspace(tmp_path: Path) -> None:
    """An invalid workspace returns an error update."""
    result = apply_changes({"workspace_path": str(tmp_path / "missing")})  # type: ignore[arg-type]
    assert result["changes"] == []
    assert result["errors"]  # type: ignore[index]
