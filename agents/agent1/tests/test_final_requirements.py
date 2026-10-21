"""Compact tests for final verification and delivery requirements."""

import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

from agents.agent1 import run_agent1
from agents.agent1.nodes.apply_changes import apply_changes
from agents.agent1.nodes.verify import verify

apply_module = importlib.import_module("agents.agent1.nodes.apply_changes")
quality_module = importlib.import_module("agents.agent1.nodes.quality_analysis")


def _verify_state(workspace: Path, snapshot: Path, tier: int, baseline_ok: bool) -> dict[str, object]:
    """Build the minimum state for a tier verification test."""
    return {
        "workspace_path": str(workspace),
        "snapshot_path": str(snapshot),
        "changes": [{"file": "main.py", "tier": tier}],
        "baseline_ok": baseline_ok,
        "baseline": {
            "command": [sys.executable, "main.py"],
            "exit_code": 0,
            "stdout": "ok\n",
            "files_changed": [],
        },
        "attempts": 0,
        "quality_before": {},
        "tool_errors": [],
    }


def test_format_only_change_passes_verification(tmp_path: Path) -> None:
    """A docstring-only Tier 1 change passes the AST-insensitive check."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("def main() -> None:\n    return None\n", encoding="utf-8")
    (workspace / "main.py").write_text('"""Docs."""\n\ndef main() -> None:\n    return None\n', encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 1, False))  # type: ignore[arg-type]
    assert result["verification_passed"] is True


def test_import_reordering_passes_ast_verification(tmp_path: Path) -> None:
    """Top-level import reordering is ignored by the AST comparison."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("import os\nimport sys\nprint('ok')\n", encoding="utf-8")
    (workspace / "main.py").write_text("import sys\nimport os\nprint('ok')\n", encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 1, False))  # type: ignore[arg-type]
    assert result["verification_passed"] is True


def test_unused_import_removal_passes_baseline_rerun(tmp_path: Path) -> None:
    """A Tier 2 unused-import removal is accepted by the baseline rerun."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("import os\nprint('ok')\n", encoding="utf-8")
    (workspace / "main.py").write_text("print('ok')\n", encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 2, True))  # type: ignore[arg-type]
    assert result["verification_passed"] is True


def test_behavior_break_is_reverted(tmp_path: Path) -> None:
    """A Tier 2 behavior change fails layer 4 and restores the snapshot."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "main.py").write_text("print('broken')\n", encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 2, True))  # type: ignore[arg-type]
    assert result["verification_passed"] is False
    assert result["verification_details"]["layer"] == 4  # type: ignore[index]
    assert result["changes"] == []
    assert (workspace / "main.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_missing_tool_is_reported(tmp_path: Path, monkeypatch: object) -> None:
    """A missing Ruff module is surfaced in tool_errors."""
    def missing_tool(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="No module named ruff")

    monkeypatch.setattr(apply_module.subprocess, "run", missing_tool)
    result = apply_changes({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert result["tool_errors"]
    assert result["tool_errors"][0]["tool"] == "ruff"  # type: ignore[index]


def test_vulture_exit_three_is_not_a_tool_error(tmp_path: Path, monkeypatch: object) -> None:
    """Vulture's findings exit code 3 is accepted as a successful tool run."""
    def tool_result(command: list[str], **kwargs: object) -> SimpleNamespace:
        if "vulture" in command:
            return SimpleNamespace(returncode=3, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(quality_module.subprocess, "run", tool_result)
    result = quality_module.quality_analysis({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert not [error for error in result["tool_errors"] if error["tool"] == "vulture"]  # type: ignore[index]


def test_original_project_is_unchanged(tmp_path: Path) -> None:
    """Running Agent 1 never modifies the supplied project."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    csv_path = project / "data.csv"
    csv_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    run_agent1(str(project), str(csv_path))
    assert {path: path.read_bytes() for path in project.rglob("*") if path.is_file()} == before
