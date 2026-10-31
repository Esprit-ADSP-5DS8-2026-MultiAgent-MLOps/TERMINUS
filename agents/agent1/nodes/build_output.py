"""Build the public Agent 1 output contract and stable artifacts."""

import difflib
import shutil
from pathlib import Path

from ..state import Agent1State


def _write_artifacts(state: Agent1State) -> tuple[str, str, list[dict[str, str]]]:
    """Copy the final workspace and write a unified snapshot diff."""
    package_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(state.get("artifact_dir") or package_root / ".workspace")
    output_path = artifact_root / "output"
    patch_path = artifact_root / "output.patch"
    errors: list[dict[str, str]] = []
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            shutil.rmtree(output_path)
        workspace_value = str(state.get("workspace_path", ""))
        workspace = Path(workspace_value) if workspace_value else None
        if workspace is not None and workspace.is_dir():
            shutil.copytree(
                workspace,
                output_path,
                ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache", "*.egg-info"),
            )
        else:
            output_path.mkdir(parents=True, exist_ok=True)
        snapshot = Path(state.get("snapshot_path", ""))
        files = set()
        if snapshot.is_dir():
            files.update(
                path.relative_to(snapshot)
                for path in snapshot.rglob("*")
                if path.is_file()
                and not any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.relative_to(snapshot).parts)
            )
        if workspace is not None and workspace.is_dir():
            files.update(
                path.relative_to(workspace)
                for path in workspace.rglob("*")
                if path.is_file()
                and not any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.relative_to(workspace).parts)
            )
        diff: list[str] = []
        for relative in sorted(files):
            before_path = snapshot / relative
            after_path = workspace / relative if workspace is not None else Path("/__agent1_missing__")
            try:
                before = before_path.read_text(encoding="utf-8").splitlines(keepends=True) if before_path.is_file() else []
                after = after_path.read_text(encoding="utf-8").splitlines(keepends=True) if after_path.is_file() else []
            except UnicodeDecodeError:
                diff.append(f"Binary files a/{relative} and b/{relative} differ\n")
                continue
            diff.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        patch_path.write_text("".join(diff), encoding="utf-8")
    except Exception as exc:
        errors.append({"node": "build_output", "message": str(exc)})
    return str(output_path), str(patch_path), errors


def _unique_errors(errors: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove repeated diagnostics produced by multiple quality passes."""
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for error in errors:
        key = repr(sorted(error.items()))
        if key not in seen:
            seen.add(key)
            unique.append(error)
    return unique


def build_output(state: Agent1State) -> dict[str, object]:
    """Return the final JSON-shaped result as an output state field."""
    before = state.get("quality_before", {})
    after = state.get("quality_after", {})
    improved_path, patch_path, artifact_errors = _write_artifacts(state)
    baseline = state.get("baseline") or {}
    output = {
        "code_quality": {
            "issues_before": int(before.get("total", 0)),
            "issues_after": int(after.get("total", 0)),
            "quality_before": before,
            "quality_after": after,
            "changes": [
                {"file": change.get("file", ""), "description": change.get("description", "")}
                for change in state.get("changes", [])
            ],
            "verified": bool(state.get("verification_passed")),
            "baseline_ok": bool(state.get("baseline_ok")),
            "verification_details": state.get("verification_details"),
            "tool_errors": _unique_errors([*state.get("tool_errors", []), *artifact_errors]),
            "failure_reason": baseline.get("failure_reason"),
            "missing_module": baseline.get("missing_module"),
            "improved_project_path": improved_path,
            "patch_path": patch_path,
            "ml_review": state.get("ml_review", {}),
            "ml_review_after": state.get("ml_review_after", {}),
            "ml_fixes": state.get("ml_fixes", {}),
            "llm": {
                "used": bool(state.get("llm", {}).get("used", False)),
                "model": state.get("llm", {}).get("model", "openai/gpt-oss-20b"),
                "endpoint_ok": bool(state.get("llm", {}).get("endpoint_ok", False)),
                "endpoint_error": state.get("llm", {}).get("endpoint_error"),
                "planned": int(state.get("llm", {}).get("planned", 0)),
                "applied": int(state.get("llm", {}).get("applied", 0)),
                "rejected": state.get("rejected", state.get("llm", {}).get("rejected", [])),
                "note": state.get("llm_note", state.get("llm", {}).get("note", "")),
            },
        },
        "problem_type": state.get("problem_type"),
        "subtype": state.get("subtype"),
        "target": state.get("target"),
        "features": state.get("features", {"numeric": [], "categorical": []}),
        "confidence": float(state.get("confidence", 0.0)),
    }
    return {"output": output}
