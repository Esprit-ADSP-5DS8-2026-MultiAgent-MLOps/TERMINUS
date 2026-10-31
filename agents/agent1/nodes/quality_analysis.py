"""Run deterministic code-quality tools and lightweight AST checks."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import ISSUE_TIER_BY_TOOL, MAX_CYCLOMATIC_COMPLEXITY
from ..state import Agent1State


def _issue(
    file: str, line: int, tool: str, code: str, message: str, tier: int | None = None
) -> dict[str, object]:
    """Build a normalized quality issue record."""
    return {
        "file": file,
        "line": line,
        "tool": tool,
        "code": code,
        "message": message,
        "tier": tier or ISSUE_TIER_BY_TOOL.get(tool, 2),
    }


def _python_files(workspace: Path) -> list[Path]:
    """Return Python files while excluding generated and environment folders."""
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache"}
    return sorted(
        path
        for path in workspace.rglob("*.py")
        if not any(part in excluded or part.endswith(".egg-info") for part in path.parts)
    )


def _run_tool(command: list[str], success_codes: set[int]) -> tuple[str, dict[str, object] | None]:
    """Run a quality tool and return stdout plus a structured tool error."""
    full_command = [sys.executable, "-m", command[0], *command[1:]]
    try:
        completed = subprocess.run(full_command, capture_output=True, text=True, check=False)
        if completed.returncode in success_codes and "No module named" not in completed.stderr:
            return completed.stdout, None
        return completed.stdout, {
            "tool": command[0],
            "command": full_command,
            "return_code": completed.returncode,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return "", {
            "tool": command[0],
            "command": full_command,
            "return_code": None,
            "stderr": str(exc),
        }


def _ruff_issues(workspace: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Parse Ruff JSON diagnostics."""
    stdout, tool_error = _run_tool(["ruff", "check", "--output-format", "json", str(workspace)], {0, 1})
    try:
        diagnostics = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return [], tool_error
    return [
        _issue(
            str(Path(item["filename"]).relative_to(workspace)),
            int(item["location"]["row"]),
            "ruff",
            str(item["code"]),
            str(item["message"]),
        )
        for item in diagnostics
        if isinstance(item, dict) and item.get("filename")
    ], tool_error


def _radon_issues(workspace: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Parse Radon cyclomatic-complexity JSON diagnostics."""
    stdout, tool_error = _run_tool(["radon", "cc", "--json", str(workspace)], {0})
    try:
        report = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return [], tool_error
    issues: list[dict[str, object]] = []
    for filename, blocks in report.items():
        for block in blocks:
            complexity = int(block.get("complexity", 0))
            if complexity > MAX_CYCLOMATIC_COMPLEXITY:
                path = Path(filename)
                relative = str(path.relative_to(workspace)) if path.is_absolute() else filename
                issues.append(
                    _issue(
                        relative,
                        int(block.get("lineno", 1)),
                        "radon",
                        "CC001",
                        f"Cyclomatic complexity is {complexity}",
                    )
                )
    return issues, tool_error


def _vulture_issues(workspace: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Parse Vulture's line-oriented dead-code output."""
    stdout, tool_error = _run_tool(["vulture", str(workspace)], {0, 3})
    issues: list[dict[str, object]] = []
    pattern = re.compile(r"^(.*?):(\d+):\s*(.*)$")
    for line in stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        issues.append(_issue(match.group(1), int(match.group(2)), "vulture", "V001", match.group(3)))
    return issues, tool_error


def _custom_issues(workspace: Path) -> list[dict[str, object]]:
    """Find missing documentation, annotations, and hardcoded filesystem paths."""
    issues: list[dict[str, object]] = []
    for path in _python_files(workspace):
        relative = str(path.relative_to(workspace))
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, SyntaxError) as exc:
            issues.append(_issue(relative, 1, "custom", "AST001", str(exc), 1))
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    issue = _issue(relative, node.lineno, "custom", "DOC001", "Missing docstring", 1)
                    issue.update({"function": node.name, "end_lineno": node.end_lineno})
                    issues.append(issue)
                if node.returns is None or any(
                    argument.annotation is None
                    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                ):
                    issue = _issue(relative, node.lineno, "custom", "ANN001", "Missing type hint", 2)
                    issue.update({"function": node.name, "end_lineno": node.end_lineno})
                    issues.append(issue)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
                    issues.append(_issue(relative, node.lineno, "custom", "PATH001", "Hardcoded absolute path", 2))
    return issues


def quality_analysis(state: Agent1State) -> dict[str, object]:
    """Analyze Python quality while converting all failures to state updates."""
    try:
        workspace = Path(state.get("workspace_path", ""))
        if not workspace.is_dir():
            raise FileNotFoundError("Workspace does not exist")
        ruff_issues, ruff_error = _ruff_issues(workspace)
        radon_issues, radon_error = _radon_issues(workspace)
        vulture_issues, vulture_error = _vulture_issues(workspace)
        grouped = {
            "ruff": ruff_issues,
            "radon": radon_issues,
            "vulture": vulture_issues,
            "custom": _custom_issues(workspace),
        }
        issues = [item for tool_issues in grouped.values() for item in tool_issues]
        counts = {tool: len(tool_issues) for tool, tool_issues in grouped.items()}
        counts["total"] = len(issues)
        return {
            "quality_issues": issues,
            "quality_before": counts,
            "tool_errors": [error for error in (ruff_error, radon_error, vulture_error) if error],
        }
    except Exception as exc:
        return {
            "quality_issues": [],
            "quality_before": {"ruff": 0, "radon": 0, "vulture": 0, "custom": 0, "total": 0},
            "tool_errors": [],
            "errors": [{"node": "quality_analysis", "message": str(exc)}],
        }
