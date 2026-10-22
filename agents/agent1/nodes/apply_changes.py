"""Apply mechanical Ruff changes and validated one-function LLM edits."""

import ast
import hashlib
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from .. import llm
from ..state import Agent1State
from .verify import normalized_ast_dump


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
        completed = subprocess.run(full_command, capture_output=True, text=True, check=False)
        if completed.returncode not in {0, 1} or "No module named" in completed.stderr:
            return {"tool": command[0], "command": full_command, "return_code": completed.returncode, "stderr": completed.stderr or completed.stdout}
    except Exception as exc:
        return {"tool": command[0], "command": full_command, "return_code": None, "stderr": str(exc)}
    return None


def _apply_step(workspace: Path, command: list[str], tier: int, description: str) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Apply one Ruff step and record only files changed by that step."""
    before = _hash_files(workspace)
    error = _run_ruff(command)
    after = _hash_files(workspace)
    changed = [name for name, digest in after.items() if before.get(name) != digest]
    return [{"file": name, "description": description, "tier": tier, "reason": description} for name in sorted(changed)], error


def _clean_answer(answer: str) -> str:
    """Remove markdown fences and surrounding prose from a model answer."""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    cleaned = fenced.group(1) if fenced else answer
    start = cleaned.find("def ")
    return cleaned[start:].strip() if start >= 0 else cleaned.strip()


def _function_node(source: str, name: str) -> ast.FunctionDef | None:
    """Find exactly one named function in a source fragment."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    matches = [node for node in functions if node.name == name]
    return matches[0] if len(functions) == 1 and len(matches) == 1 and isinstance(matches[0], ast.FunctionDef) else None


def _parameter_names(node: ast.FunctionDef) -> list[str]:
    """Return positional, vararg, keyword-only, and kwarg names in order."""
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    names = [arg.arg for arg in args]
    if node.args.vararg:
        names.append(node.args.vararg.arg)
    if node.args.kwarg:
        names.append(node.args.kwarg.arg)
    return names


def _has_type_hints(node: ast.FunctionDef) -> bool:
    """Return whether every parameter and the return value is annotated."""
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return node.returns is not None and all(arg.annotation is not None for arg in args)


def _validate_answer(original: str, answer: str, name: str, kind: str) -> tuple[bool, str]:
    """Validate model syntax, identity, behavior, requested fix, comments, and size."""
    node = _function_node(answer, name)
    original_node = _function_node(original, name)
    if node is None or original_node is None:
        return False, "answer must contain exactly one matching FunctionDef"
    if _parameter_names(node) != _parameter_names(original_node):
        return False, "parameter names or order changed"
    if normalized_ast_dump(original, remove_annotations=True) != normalized_ast_dump(answer, remove_annotations=True):
        return False, "function logic changed"
    wants_docstring = kind in {"missing_docstring", "missing_both"}
    wants_hints = kind in {"missing_type_hints", "missing_both"}
    if wants_docstring and ast.get_docstring(node) is None:
        return False, "requested docstring was not added"
    if wants_hints and not _has_type_hints(node):
        return False, "requested type hints were not added"
    if answer.count("\n#") + (1 if answer.startswith("#") else 0) < original.count("\n#") + (1 if original.startswith("#") else 0):
        return False, "comment lines were lost"
    if len(answer.splitlines()) > max(1, len(original.splitlines()) * 3):
        return False, "answer is longer than three times the original"
    return True, ""


def _apply_llm_item(workspace: Path, item: dict[str, object], client: object) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    """Apply one validated function replacement."""
    relative = Path(str(item["file"]))
    path = workspace / relative
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        target = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == item["function"])
        original = textwrap.dedent("\n".join(source.splitlines()[target.lineno - 1 : target.end_lineno]))
        issue = llm.ISSUES[str(item["kind"])]
        prompt = llm.USER.format(issue=issue, function_source=original)
        answer = _clean_answer(client.complete_text(llm.SYSTEM, prompt))
        valid, reason = _validate_answer(original, answer, str(item["function"]), str(item["kind"]))
        if not valid:
            return None, {"function": str(item["function"]), "reason": reason}
        indent = re.match(r"^\s*", source.splitlines()[target.lineno - 1]).group(0)
        replacement = "\n".join(indent + line if line else line for line in answer.splitlines())
        lines = source.splitlines(keepends=True)
        replacement += "\n" if lines[target.end_lineno - 1].endswith("\n") else ""
        lines[target.lineno - 1 : target.end_lineno] = [replacement]
        path.write_text("".join(lines), encoding="utf-8")
        return {"file": str(relative), "function": str(item["function"]), "description": f"LLM applied {item['kind']}", "tier": 2, "reason": "Validated function-only edit"}, None
    except Exception as exc:
        return None, {"function": str(item.get("function", "")), "reason": str(exc)}


def apply_changes(state: Agent1State) -> dict[str, object]:
    """Apply Ruff changes followed by validated per-function LLM edits."""
    try:
        workspace = Path(state.get("workspace_path", ""))
        if not workspace.is_dir():
            raise FileNotFoundError("Workspace does not exist")
        changes: list[dict[str, object]] = []
        tool_errors: list[dict[str, object]] = []
        for command, description in ((["ruff", "format", str(workspace)], "Tier 1 Ruff formatting"), (["ruff", "check", "--select", "I", "--fix", str(workspace)], "Tier 1 Ruff import sorting")):
            step_changes, error = _apply_step(workspace, command, 1, description)
            changes.extend(step_changes)
            if error:
                tool_errors.append(error)
        if state.get("baseline_ok") and 2 in state.get("allowed_tiers", []):
            step_changes, error = _apply_step(workspace, ["ruff", "check", "--fix", str(workspace)], 2, "Tier 2 Ruff safe fixes")
            changes.extend(step_changes)
            if error:
                tool_errors.append(error)
        rejected: list[dict[str, str]] = []
        llm_used = False
        client = state.get("llm_client") or llm.get_client()
        by_file: dict[str, list[dict[str, object]]] = {}
        for item in state.get("improvement_plan", []):
            by_file.setdefault(str(item["file"]), []).append(item)
        for items in by_file.values():
            for item in sorted(items, key=lambda value: int(value["line_range"][0]), reverse=True):
                applied, rejected_item = _apply_llm_item(workspace, item, client)
                if applied:
                    changes.append(applied)
                    llm_used = True
                if rejected_item:
                    rejected.append(rejected_item)
        changed_files = sorted({str(change["file"]) for change in changes if change.get("tier") == 2})
        if changed_files:
            _run_ruff(["ruff", "format", *[str(workspace / file) for file in changed_files]])
        llm_state = dict(state.get("llm", {}))
        llm_state.update({"used": llm_used, "applied": sum(1 for change in changes if change.get("reason") == "Validated function-only edit"), "rejected": rejected})
        return {"changes": changes, "rejected": rejected, "llm": llm_state, "tool_errors": [*state.get("tool_errors", []), *tool_errors]}
    except Exception as exc:
        return {"changes": [], "errors": [{"node": "apply_changes", "message": str(exc)}]}
