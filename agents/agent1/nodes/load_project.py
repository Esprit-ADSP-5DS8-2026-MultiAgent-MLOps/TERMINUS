"""Copy an input project into isolated working directories."""

import shutil
import stat
import tempfile
from pathlib import Path

from ..state import Agent1State


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    """Return generated and environment directories excluded from a copy."""
    ignored = {".venv", ".git", "__pycache__", ".pytest_cache"}
    return {name for name in names if name in ignored or name.endswith(".egg-info")}


def load_project(state: Agent1State) -> dict[str, object]:
    """Copy the source project into a workspace and an untouched snapshot."""
    try:
        source = Path(state["project_path"]).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Project directory not found: {source}")
        root = Path(tempfile.mkdtemp(prefix="agent1-"))
        workspace = root / "workspace"
        snapshot = root / "snapshot"
        shutil.copytree(source, workspace, ignore=_copy_ignore)
        shutil.copytree(source, snapshot, ignore=_copy_ignore)
        # The orchestrator gives a read-only copy of the validated project (chmod 444):
        # the workspace must be writable for ruff and the edits.
        for path in [workspace, *workspace.rglob("*")]:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        return {
            "workspace_path": str(workspace),
            "snapshot_path": str(snapshot),
        }
    except Exception as exc:
        return {
            "workspace_path": "",
            "snapshot_path": "",
            "errors": [{"node": "load_project", "message": str(exc)}],
        }
