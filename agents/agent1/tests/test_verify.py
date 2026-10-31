"""Tests for layered verification and rollback."""

from pathlib import Path

from agents.agent1.nodes.revert_and_continue import revert_and_continue
from agents.agent1.nodes.verify import verify


def _state(workspace: Path, snapshot: Path) -> dict[str, object]:
    return {
        "workspace_path": str(workspace),
        "snapshot_path": str(snapshot),
        "baseline_ok": False,
        "baseline": None,
        "changes": [{"file": "main.py", "tier": 1}],
        "attempts": 0,
    }


def test_verify_passes_compile_and_import(tmp_path: Path) -> None:
    """Valid workspace modules pass the non-baseline verification layers."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    source = "def main() -> None:\n    return None\n"
    (workspace / "main.py").write_text(source, encoding="utf-8")
    (snapshot / "main.py").write_text(source, encoding="utf-8")
    result = verify(_state(workspace, snapshot))  # type: ignore[arg-type]
    assert result["verification_passed"] is True
    assert "quality_after" in result


def test_verify_restores_snapshot_on_compile_failure(tmp_path: Path) -> None:
    """A failed compile restores the changed workspace file and increments attempts."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("print('original')\n", encoding="utf-8")
    (workspace / "main.py").write_text("def broken(:\n", encoding="utf-8")
    result = verify(_state(workspace, snapshot))  # type: ignore[arg-type]
    assert result["verification_passed"] is False
    assert result["attempts"] == 1
    assert (workspace / "main.py").read_text(encoding="utf-8") == "print('original')\n"


def test_revert_and_continue_restores_snapshot(tmp_path: Path) -> None:
    """The terminal recovery node restores the snapshot."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("print('original')\n", encoding="utf-8")
    (workspace / "main.py").write_text("print('changed')\n", encoding="utf-8")
    result = revert_and_continue(_state(workspace, snapshot))  # type: ignore[arg-type]
    assert result["changes"] == []
    assert (workspace / "main.py").read_text(encoding="utf-8") == "print('original')\n"
