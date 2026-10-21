"""Tests for project workspace creation."""

from pathlib import Path

from agents.agent1.nodes.load_project import load_project


def test_load_project_creates_workspace_and_snapshot(tmp_path: Path) -> None:
    """The source remains unchanged while two filtered copies are created."""
    source = tmp_path / "project"
    source.mkdir()
    (source / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "secret.txt").write_text("ignored", encoding="utf-8")
    (source / "module.egg-info").mkdir()
    (source / "module.egg-info" / "metadata").write_text("ignored", encoding="utf-8")

    result = load_project({"project_path": str(source)})  # type: ignore[arg-type]

    workspace = Path(str(result["workspace_path"]))
    snapshot = Path(str(result["snapshot_path"]))
    assert not result.get("errors")
    assert (workspace / "main.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert (snapshot / "main.py").exists()
    assert not (workspace / ".venv").exists()
    assert not (snapshot / "module.egg-info").exists()
    assert (source / "main.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_load_project_records_errors(tmp_path: Path) -> None:
    """Invalid input is reported instead of raised."""
    result = load_project({"project_path": str(tmp_path / "missing")})  # type: ignore[arg-type]
    assert result["workspace_path"] == ""
    assert result["snapshot_path"] == ""
    assert result["errors"]
