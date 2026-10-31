"""Discover and execute a project's baseline entry point."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ..config import BASELINE_OUTPUT_MAX_CHARS, BASELINE_TIMEOUT_SECONDS
from ..state import Agent1State


def _read_manifest(workspace: Path) -> dict[str, Any]:
    """Read the optional project manifest, returning an empty mapping on error."""
    manifest_path = workspace / "mlops.yaml"
    if not manifest_path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def find_entry_point(workspace_path: str) -> tuple[list[str] | None, str | None]:
    """Find a manifest run command or a conventional Python entry point."""
    workspace = Path(workspace_path)
    manifest = _read_manifest(workspace)
    declared = manifest.get("run") or manifest.get("command") or manifest.get("entrypoint")
    if not declared and isinstance(manifest.get("entrypoints"), dict):
        declared = manifest["entrypoints"].get("run") or manifest["entrypoints"].get("main")
    if isinstance(declared, str) and declared.strip():
        command = shlex.split(declared)
        if command and Path(command[0]).name in {"python", "python3"}:
            command[0] = sys.executable
        elif command and command[0].endswith(".py"):
            command.insert(0, sys.executable)
        return command, declared

    for name in ("main.py", "train.py", "app.py", "run.py"):
        candidate = workspace / name
        if candidate.is_file():
            return [sys.executable, name], name
    for candidate in sorted(workspace.rglob("*.py")):
        try:
            source = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source:
            return [sys.executable, str(candidate.relative_to(workspace))], str(
                candidate.relative_to(workspace)
            )
    return None, None


def _fingerprint_files(workspace: Path) -> dict[str, str]:
    """Hash workspace files so created and modified outputs can be identified."""
    fingerprints: dict[str, str] = {}
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in relative.parts):
            continue
        try:
            fingerprints[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return fingerprints


def _failure_update(message: str) -> dict[str, object]:
    """Return the common state update for a failed baseline."""
    return {
        "baseline": {"error": message},
        "baseline_ok": False,
        "allowed_tiers": [1],
        "errors": [{"node": "capture_baseline", "message": message}],
    }


def _failure_fields(stderr: str) -> dict[str, str]:
    """Classify a baseline failure and extract a missing dependency name."""
    match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)", stderr)
    if match:
        return {"failure_reason": "missing_dependency", "missing_module": match.group(1)}
    return {}


def capture_baseline(state: Agent1State) -> dict[str, object]:
    """Run the discovered entry point and record its baseline behavior."""
    workspace = Path(state.get("workspace_path", ""))
    if not workspace.is_dir():
        return _failure_update("Workspace does not exist")
    command, entry_point = find_entry_point(str(workspace))
    if command is None:
        return _failure_update("No entry point found")
    before = _fingerprint_files(workspace)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=BASELINE_TIMEOUT_SECONDS,
            check=False,
        )
        after = _fingerprint_files(workspace)
        changed = sorted(
            name for name, digest in after.items() if before.get(name) != digest
        )
        changed.extend(sorted(set(after) - set(before)))
        baseline = {
            "entry_point": entry_point,
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[:BASELINE_OUTPUT_MAX_CHARS],
            "stderr": completed.stderr[:BASELINE_OUTPUT_MAX_CHARS],
            "files_changed": sorted(set(changed)),
        }
        baseline.update(_failure_fields(completed.stderr))
        if completed.returncode != 0:
            return {
                "baseline": baseline,
                "baseline_ok": False,
                "allowed_tiers": [1],
            }
        return {"baseline": baseline, "baseline_ok": True, "allowed_tiers": [1, 2, 3]}
    except subprocess.TimeoutExpired as exc:
        return _failure_update(f"Baseline timed out: {exc}")
    except Exception as exc:
        return _failure_update(str(exc))
