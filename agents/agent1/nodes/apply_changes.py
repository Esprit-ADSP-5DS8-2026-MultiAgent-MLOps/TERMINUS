"""Apply only mechanical Tier 1 Ruff changes."""

import hashlib
import subprocess
import sys
from pathlib import Path

from ..state import Agent1State


def _hash_files(workspace: Path) -> dict[str, str]:
    """Hash Python files in the workspace."""
    return {
        str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*.py")
        if path.is_file()
    }


def _run_ruff(command: list[str]) -> dict[str, object] | None:
    """Run one Ruff command and return a structured error, if any."""
    full_command = [sys.executable, "-m", command[0], *command[1:]]
    try:
        completed = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode not in {0, 1} or "No module named" in completed.stderr:
            return {
                "tool": command[0],
                "command": full_command,
                "return_code": completed.returncode,
                "stderr": completed.stderr or completed.stdout,
            }
    except Exception as exc:
        return {
            "tool": command[0],
            "command": full_command,
            "return_code": None,
            "stderr": str(exc),
        }
    return None


def _apply_step(
    workspace: Path,
    command: list[str],
    tier: int,
    description: str,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Apply one Ruff step and record only files changed by that step."""
    before = _hash_files(workspace)
    error = _run_ruff(command)
    after = _hash_files(workspace)
    changed = [name for name, digest in after.items() if before.get(name) != digest]
    return [
        {
            "file": name,
            "description": description,
            "tier": tier,
            "reason": description,
        }
        for name in sorted(changed)
    ], error


def apply_changes(state: Agent1State) -> dict[str, object]:
    """Run Ruff's safe fixes and record every changed Python file as Tier 1."""
    try:
        workspace = Path(state.get("workspace_path", ""))
        if not workspace.is_dir():
            raise FileNotFoundError("Workspace does not exist")
        updates: list[dict[str, object]] = []
        tool_errors: list[dict[str, object]] = []
        tier_one_steps = (
            (["ruff", "format", str(workspace)], "Tier 1 Ruff formatting"),
            (["ruff", "check", "--select", "I", "--fix", str(workspace)], "Tier 1 Ruff import sorting"),
        )
        for command, description in tier_one_steps:
            step_changes, error = _apply_step(workspace, command, 1, description)
            updates.extend(step_changes)
            if error:
                tool_errors.append(error)
        if state.get("baseline_ok") and 2 in state.get("allowed_tiers", []):
            step_changes, error = _apply_step(
                workspace,
                ["ruff", "check", "--fix", str(workspace)],
                2,
                "Tier 2 Ruff safe fixes",
            )
            updates.extend(step_changes)
            if error:
                tool_errors.append(error)
        result: dict[str, object] = {
            "changes": updates,
            "tool_errors": [*state.get("tool_errors", []), *tool_errors],
        }
        return result
    except Exception as exc:
        return {
            "changes": [],
            "errors": [{"node": "apply_changes", "message": str(exc)}],
        }
