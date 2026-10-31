"""Verify behavior-changing ML fixes without comparing metrics."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import BASELINE_OUTPUT_MAX_CHARS, BASELINE_TIMEOUT_SECONDS
from .ml_review import ml_review
from .ml_fix import _diff


def verify_ml(state: dict[str, Any]) -> dict[str, Any]:
    """Compile, import, rerun baseline, and restore ML fixes on failure."""
    workspace = Path(state.get("workspace_path", ""))
    snapshot = Path(state.get("pre_ml_snapshot", ""))
    details = state.get("ml_fixes", {})
    failure = ""
    try:
        compiled = subprocess.run([sys.executable, "-m", "compileall", "-q", str(workspace)], capture_output=True, text=True, check=False)
        if compiled.returncode != 0:
            failure = f"compile failed: {compiled.stderr[:BASELINE_OUTPUT_MAX_CHARS]}"
        baseline = state.get("baseline") or {}
        if not failure and state.get("baseline_ok") and baseline.get("command"):
            run = subprocess.run(baseline["command"], cwd=workspace, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True, timeout=BASELINE_TIMEOUT_SECONDS, check=False)
            details["stdout_after"] = run.stdout[:1500]
            if run.returncode != 0:
                failure = f"baseline rerun failed with exit code {run.returncode}"
            before_files = sorted(baseline.get("files_changed", []))
            after_files = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file() and not any(part == "__pycache__" for part in p.parts))
            if not failure and before_files and not set(before_files).issubset(set(after_files)):
                failure = "baseline output files differ"
        if failure:
            shutil.rmtree(workspace)
            shutil.copytree(snapshot, workspace)
            details["applied"] = []
            details["behavior_changing"] = False
            details["note"] = f"ML fixes reverted: {failure}"
            return {"ml_fixes": details, "behavior_changing": False, "ml_review_after": ml_review({**state, "workspace_path": str(workspace)})["ml_review"]}
        details["verified"] = True
        patch_path = Path(state.get("artifact_dir") or Path(__file__).resolve().parents[1] / ".workspace") / "ml_output.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(_diff(snapshot, workspace), encoding="utf-8")
        details["ml_patch_path"] = str(patch_path)
        review = ml_review({**state, "workspace_path": str(workspace)})
        return {"ml_fixes": details, "behavior_changing": bool(state.get("behavior_changing")), "ml_review_after": review["ml_review"], "tool_errors": [*state.get("tool_errors", []), *review.get("tool_errors", [])]}
    except Exception as exc:
        if snapshot.is_dir():
            shutil.rmtree(workspace)
            shutil.copytree(snapshot, workspace)
        details["applied"] = []
        details["behavior_changing"] = False
        details["note"] = f"ML fixes reverted: {exc}"
        return {"ml_fixes": details, "behavior_changing": False, "ml_review_after": {"high": 0, "medium": 0, "low": 0, "findings": []}}
