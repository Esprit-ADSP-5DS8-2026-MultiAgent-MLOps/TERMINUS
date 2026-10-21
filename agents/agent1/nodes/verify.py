"""Verify changes in layers and restore the snapshot on failure."""

import ast
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import BASELINE_OUTPUT_MAX_CHARS, BASELINE_TIMEOUT_SECONDS, OUTPUT_TOLERANCE
from .quality_analysis import quality_analysis
from ..state import Agent1State


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstring expressions before comparing two syntax trees."""

    def _strip(self, node: ast.AST) -> ast.AST:
        """Remove a leading docstring from a body and visit descendants."""
        if hasattr(node, "body") and isinstance(node.body, list):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
        return self.generic_visit(node)

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


class _TopLevelImportSorter(ast.NodeTransformer):
    """Sort top-level imports while preserving all non-import statements."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Sort module-level Import and ImportFrom nodes."""
        imports = [item for item in node.body if isinstance(item, (ast.Import, ast.ImportFrom))]
        if imports:
            ordered = sorted(imports, key=ast.unparse)
            positions = [index for index, item in enumerate(node.body) if isinstance(item, (ast.Import, ast.ImportFrom))]
            for index, item in zip(positions, ordered):
                node.body[index] = item
        return self.generic_visit(node)


def _syntax_tree(path: Path) -> str:
    """Return an AST representation ignoring docstrings and import order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = _DocstringStripper().visit(tree)
    tree = _TopLevelImportSorter().visit(tree)
    return ast.dump(tree, annotate_fields=True)


def _ast_diff(current: Path, original: Path) -> str:
    """Return a truncated unified diff for an AST mismatch."""
    diff = difflib.unified_diff(
        original.read_text(encoding="utf-8").splitlines(keepends=True),
        current.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile=f"snapshot/{original.name}",
        tofile=f"workspace/{current.name}",
    )
    return "".join(diff)[:BASELINE_OUTPUT_MAX_CHARS]


def _fingerprint_files(workspace: Path) -> dict[str, str]:
    """Hash all workspace files for output comparison and rollback."""
    result: dict[str, str] = {}
    for path in workspace.rglob("*"):
        if path.is_file():
            result[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def restore_snapshot(workspace_path: str, snapshot_path: str) -> None:
    """Restore the workspace contents exactly from the untouched snapshot."""
    workspace = Path(workspace_path)
    snapshot = Path(snapshot_path)
    if not workspace.is_dir() or not snapshot.is_dir():
        return
    for path in sorted(workspace.rglob("*"), reverse=True):
        relative = path.relative_to(workspace)
        if not (snapshot / relative).exists():
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
    shutil.copytree(snapshot, workspace, dirs_exist_ok=True)


def _restore_files(workspace: Path, snapshot: Path, filenames: set[str]) -> None:
    """Restore only affected files from the untouched snapshot."""
    for filename in filenames:
        current = workspace / filename
        original = snapshot / filename
        if original.is_file():
            current.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, current)
        elif current.exists():
            current.unlink()


def _import_modules(workspace: Path) -> str | None:
    """Import each Python module in an isolated subprocess."""
    modules = [
        str(path.relative_to(workspace).with_suffix("")).replace(os.sep, ".")
        for path in workspace.rglob("*.py")
        if path.name != "__init__.py"
    ]
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=workspace,
            env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(workspace)},
            capture_output=True,
            text=True,
            timeout=BASELINE_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            return f"Import failed for {module}: {completed.stderr[:BASELINE_OUTPUT_MAX_CHARS]}"
    return None


def _stdout_matches(before: str, after: str) -> bool:
    """Compare stdout structure and numeric values within configured tolerance."""
    number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    before_numbers = [float(value) for value in number_pattern.findall(before)]
    after_numbers = [float(value) for value in number_pattern.findall(after)]
    if len(before_numbers) != len(after_numbers):
        return False
    if any(abs(left - right) > OUTPUT_TOLERANCE for left, right in zip(before_numbers, after_numbers)):
        return False
    return number_pattern.sub("<number>", before).split() == number_pattern.sub(
        "<number>", after
    ).split()


def _rerun_matches(state: Agent1State, workspace: Path) -> str | None:
    """Rerun the baseline command and compare exit code, output files, and stdout."""
    baseline = state.get("baseline") or {}
    command = baseline.get("command")
    if not isinstance(command, list):
        return "Baseline command is missing"
    expected = sorted(baseline.get("files_changed", []))
    snapshot = Path(state.get("snapshot_path", ""))
    for name in expected:
        generated = workspace / name
        if not (snapshot / name).exists() and generated.is_file():
            generated.unlink()
    before = _fingerprint_files(workspace)
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env={**os.environ, "PYTHONHASHSEED": "0"},
            capture_output=True,
            text=True,
            timeout=BASELINE_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception as exc:
        return str(exc)
    after = _fingerprint_files(workspace)
    changed = sorted(name for name, digest in after.items() if before.get(name) != digest)
    if completed.returncode != baseline.get("exit_code"):
        return "Baseline exit code changed"
    if changed != expected:
        return f"Output files changed: expected {expected}, got {changed}"
    if not _stdout_matches(str(baseline.get("stdout", "")), completed.stdout[:BASELINE_OUTPUT_MAX_CHARS]):
        return "Baseline stdout structure or metrics changed"
    return None


def verify(state: Agent1State) -> dict[str, object]:
    """Run layered verification and restore the snapshot after any failure."""
    if not state.get("changes"):
        quality = quality_analysis(state) if Path(state.get("workspace_path", "")).is_dir() else {}
        return {
            "verification_passed": True,
            "verification_details": {
                "layer": 0,
                "name": "no_changes",
                "message": "No changes were applied; verification is a successful no-op.",
            },
            "quality_after": quality.get("quality_before", state.get("quality_before", {})),
            "quality_issues": quality.get("quality_issues", state.get("quality_issues", [])),
            "tool_errors": [*state.get("tool_errors", []), *quality.get("tool_errors", [])],
        }
    workspace = Path(state.get("workspace_path", ""))
    failure: str | None = None
    failure_layer = 0
    failure_name = ""
    verification_warning: dict[str, object] | None = None
    try:
        if not workspace.is_dir():
            failure = "Workspace does not exist"
            failure_layer, failure_name = 1, "compile"
        else:
            compiled = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", str(workspace)],
                capture_output=True,
                text=True,
                check=False,
            )
            if compiled.returncode != 0:
                failure = f"Compilation failed: {compiled.stderr[:BASELINE_OUTPUT_MAX_CHARS]}"
                failure_layer, failure_name = 1, "compile"
            if failure is None:
                failure = _import_modules(workspace)
                if failure:
                    failure_layer, failure_name = 2, "imports"
            if failure is None:
                snapshot = Path(state.get("snapshot_path", ""))
                tier_one_changes = [change for change in state.get("changes", []) if change.get("tier") == 1]
                for change in tier_one_changes:
                    relative = Path(str(change.get("file", "")))
                    current = workspace / relative
                    original = snapshot / relative
                    if current.is_file() and original.is_file() and _syntax_tree(current) != _syntax_tree(original):
                        message = f"AST changed for Tier 1 file: {relative}"
                        detail = {
                            "layer": 3,
                            "name": "AST comparison",
                            "message": message,
                            "diff": _ast_diff(current, original),
                        }
                        if state.get("baseline_ok"):
                            verification_warning = {**detail, "severity": "warning"}
                        else:
                            failure = message
                            failure_layer, failure_name = 3, "AST comparison"
                            verification_warning = detail
                        break
            rerun_needed = bool(state.get("baseline_ok")) and (
                any(change.get("tier") == 2 for change in state.get("changes", []))
                or verification_warning is not None
            )
            if failure is None and rerun_needed:
                failure = _rerun_matches(state, workspace)
                if failure:
                    failure_layer, failure_name = 4, "baseline rerun"
        if failure:
            failed_tier = 2 if failure_layer == 4 and any(change.get("tier") == 2 for change in state.get("changes", [])) else 1
            affected_files = {
                str(change.get("file"))
                for change in state.get("changes", [])
                if change.get("tier") == failed_tier
            }
            snapshot = Path(state.get("snapshot_path", ""))
            _restore_files(workspace, snapshot, affected_files)
            remaining_changes = [
                change
                for change in state.get("changes", [])
                if str(change.get("file")) not in affected_files
            ]
            quality = quality_analysis({**state, "workspace_path": str(workspace)})
            return {
                "verification_passed": False,
                "attempts": state.get("attempts", 0) + 1,
                "changes": remaining_changes,
                "verification_details": {
                    "layer": failure_layer,
                    "name": failure_name,
                    "message": failure,
                    **({"warning": verification_warning} if verification_warning else {}),
                    **({"diff": verification_warning["diff"]} if verification_warning and "diff" in verification_warning else {}),
                },
                "quality_after": quality.get("quality_before", {}),
                "quality_issues": quality.get("quality_issues", []),
                "tool_errors": [*state.get("tool_errors", []), *quality.get("tool_errors", [])],
                "errors": [{"node": "verify", "message": failure}],
            }
        quality = quality_analysis({**state, "workspace_path": str(workspace)})
        return {
            "verification_passed": True,
            "verification_details": {
                "layer": None,
                "name": "complete",
                "message": "All applicable verification layers passed.",
                **({"warning": verification_warning} if verification_warning else {}),
            },
            "quality_after": quality.get("quality_before", {}),
            "quality_issues": quality.get("quality_issues", []),
            "tool_errors": [*state.get("tool_errors", []), *quality.get("tool_errors", [])],
        }
    except Exception as exc:
        affected_files = {str(change.get("file")) for change in state.get("changes", [])}
        _restore_files(workspace, Path(state.get("snapshot_path", "")), affected_files)
        quality = quality_analysis({**state, "workspace_path": str(workspace)}) if workspace.is_dir() else {}
        return {
            "verification_passed": False,
            "attempts": state.get("attempts", 0) + 1,
            "changes": [],
            "verification_details": {
                "layer": failure_layer or 1,
                "name": failure_name or "unexpected error",
                "message": str(exc),
            },
            "quality_after": quality.get("quality_before", {}),
            "quality_issues": quality.get("quality_issues", []),
            "tool_errors": [*state.get("tool_errors", []), *quality.get("tool_errors", [])],
            "errors": [{"node": "verify", "message": str(exc)}],
        }
