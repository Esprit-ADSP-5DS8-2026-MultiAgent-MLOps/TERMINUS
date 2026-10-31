"""Opt-in behavior-changing ML fix stage."""

import ast
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
import textwrap

from ..config import ML_SEED
from .. import llm
from .ml_review import ml_review

ML_SYSTEM = "You are an ML code reviewer; fix ONLY the described problem; keep the function name, signature and return value; prefer an sklearn Pipeline or splitting before fitting; change nothing else. Return JSON with imports and code."


def _replace_calls(source: str, findings: list[dict[str, Any]], classification: bool) -> tuple[str, list[dict[str, Any]]]:
    """Apply safe call-argument fixes using AST line spans."""
    tree = ast.parse(source)
    active_rules = {str(finding["rule"]) for finding in findings}
    edits: list[tuple[int, int, int, int, str, dict[str, Any]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        rules: list[str] = []
        if name == "train_test_split":
            if "NO_RANDOM_STATE" in active_rules and not any(k.arg == "random_state" for k in node.keywords):
                rules.append("NO_RANDOM_STATE")
                node.keywords.append(ast.keyword(arg="random_state", value=ast.Constant(ML_SEED)))
            if "NO_STRATIFY" in active_rules and classification and not any(k.arg == "stratify" for k in node.keywords) and len(node.args) > 1:
                rules.append("NO_STRATIFY")
                node.keywords.append(ast.keyword(arg="stratify", value=node.args[1]))
        supported = {"LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier", "SVC", "LinearSVC", "SGDClassifier", "GradientBoostingClassifier", "GradientBoostingRegressor"}
        if "NO_RANDOM_STATE" in active_rules and name in supported and not any(k.arg == "random_state" for k in node.keywords):
            rules.append("NO_RANDOM_STATE")
            node.keywords.append(ast.keyword(arg="random_state", value=ast.Constant(ML_SEED)))
        if "IMBALANCE_NOT_HANDLED" in active_rules and name in supported and not any(k.arg == "class_weight" for k in node.keywords):
            rules.append("IMBALANCE_NOT_HANDLED")
            node.keywords.append(ast.keyword(arg="class_weight", value=ast.Constant("balanced")))
        if rules:
            original = ast.get_source_segment(source, node)
            replacement = ast.unparse(node)
            if original:
                edits.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset, replacement, {"rules": rules, "line": node.lineno}))
    lines = source.splitlines(keepends=True)
    applied = []
    for start, end, col_start, col_end, replacement, finding in sorted(edits, reverse=True):
        if start == end:
            line = lines[start - 1]
            lines[start - 1] = line[:col_start] + replacement + line[col_end:]
        else:
            suffix = "\n" if lines[end - 1].endswith("\n") else ""
            indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
            replacement = "\n".join(indent + line if line else line for line in replacement.splitlines()) + suffix
            lines[start - 1:end] = [replacement]
        descriptions = {"NO_RANDOM_STATE": "Added random_state=42", "NO_STRATIFY": "Added stratify using the target expression", "IMBALANCE_NOT_HANDLED": "Added class_weight=balanced"}
        for rule in finding["rules"]:
            applied.append({"rule": rule, "line": finding["line"], "method": "deterministic", "description": descriptions[rule]})
    return "".join(lines), applied


def _diff(before: Path, after: Path) -> str:
    """Create a unified diff between two trees (Python files only)."""
    def py_files(root: Path) -> set[Path]:
        return {p.relative_to(root) for p in root.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts}

    output: list[str] = []
    for relative in sorted(py_files(before) | py_files(after)):
        lp, rp = before / relative, after / relative
        left = lp.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if lp.is_file() else []
        right = rp.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if rp.is_file() else []
        output.extend(difflib.unified_diff(left, right, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    return "".join(output)


def _entry_command(state: dict[str, Any]) -> list[str] | None:
    """Build the command that runs the project's entry point."""
    baseline = state.get("baseline") or {}
    command = baseline.get("command")
    if command:
        return list(command)
    entry = baseline.get("entry_point") or baseline.get("entry")
    return [sys.executable, str(entry)] if entry else None


def _run_candidate(state: dict[str, Any], workspace: Path) -> tuple[bool, str, str]:
    """Compile and run the entry point; return (success, stderr, stdout), both truncated."""
    compiled = subprocess.run([sys.executable, "-m", "compileall", "-q", str(workspace)], capture_output=True, text=True, check=False)
    if compiled.returncode != 0:
        return False, (compiled.stdout + compiled.stderr)[:1500], ""
    command = _entry_command(state)
    if not command:
        return True, "", ""
    try:
        result = subprocess.run(command, cwd=workspace, env={**os.environ, "PYTHONHASHSEED": "0"},
                                capture_output=True, text=True, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "timeout", ""
    return result.returncode == 0, result.stderr[:1500], result.stdout[:1500]


LLM_RULES = {"DATA_LEAKAGE", "WRONG_METRIC"}

LLM_SYSTEM = (
    "You are an ML code reviewer. Fix ONLY the described problem in the given Python function. "
    "Keep the function name, its parameters and what it returns. Keep every existing argument "
    "such as random_state and stratify. Fit preprocessing (scalers, encoders, imputers) only on "
    "the training data AFTER train_test_split, or use an sklearn Pipeline. Change nothing else. "
    'Answer with JSON only, no markdown: {"imports": ["import lines needed"], "code": "the full replacement function"}.'
)


def _current_findings(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-run the ML review on the current workspace."""
    try:
        return list((ml_review(state).get("ml_review") or {}).get("findings", []))
    except Exception:
        return []


def _rule_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    """Count findings per rule."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[str(f["rule"])] = counts.get(str(f["rule"]), 0) + 1
    return counts


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Parse a JSON answer, tolerating markdown fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("answer is not JSON")
    return json.loads(text[start:end + 1])


def _arg_names(fn: ast.FunctionDef) -> list[str]:
    """Parameter names of a function, in order."""
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] + [x.arg for x in (a.vararg, a.kwarg) if x]


def _enclosing_function(tree: ast.AST, line: int) -> ast.FunctionDef | None:
    """Innermost function that contains the given line."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno <= line <= (node.end_lineno or node.lineno):
            if best is None or node.lineno >= best.lineno:
                best = node
    return best


def _replace_function(source: str, fn: ast.FunctionDef, new_code: str) -> str:
    """Replace the lines of one function with new code."""
    lines = source.splitlines(keepends=True)
    start = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
    block = textwrap.indent(new_code.rstrip("\n"), " " * fn.col_offset) + "\n"
    lines[start - 1:fn.end_lineno] = [block]
    return "".join(lines)


def _add_imports(source: str, import_lines: list[str]) -> str:
    """Add missing import lines after the last top-level import."""
    tree = ast.parse(source)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    existing = {ast.unparse(n) for n in imports}
    new: list[str] = []
    for line in import_lines:
        node = ast.parse(line.strip()).body[0]
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports must be plain import lines")
        text = ast.unparse(node)
        if text not in existing and text not in new:
            new.append(text)
    if not new:
        return source
    last = max((n.end_lineno or 0 for n in imports), default=0)
    lines = source.splitlines(keepends=True)
    lines[last:last] = [t + "\n" for t in new]
    return "".join(lines)


def _build_candidate(source: str, fn: ast.FunctionDef, answer: str) -> str:
    """Build the candidate file from the LLM answer, or raise ValueError."""
    data = _parse_llm_json(answer)
    code = textwrap.dedent(str(data.get("code", ""))).strip("\n")
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    hoisted = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if len(funcs) != 1 or len(funcs) + len(hoisted) != len(tree.body):
        raise ValueError("answer must contain exactly one function")
    new_fn = funcs[0]
    if new_fn.name != fn.name or _arg_names(new_fn) != _arg_names(fn):
        raise ValueError("function name or parameters changed")
    start = min([d.lineno for d in new_fn.decorator_list] + [new_fn.lineno])
    new_code = "\n".join(code.splitlines()[start - 1:new_fn.end_lineno])
    old_len = len(ast.get_source_segment(source, fn) or "")
    if old_len and len(new_code) > 3 * old_len:
        raise ValueError("answer is more than 3x longer than the original")
    out = _replace_function(source, fn, new_code)
    out = _add_imports(out, [str(x) for x in (data.get("imports") or [])] + [ast.unparse(n) for n in hoisted])
    ast.parse(out)
    return out


def _llm_stage(state: dict[str, Any], workspace: Path, applied: list, rejected: list) -> tuple[str, str]:
    """Try LLM fixes one at a time. Returns (note, last_stdout)."""
    targets = [f for f in _current_findings(state) if str(f.get("rule")) in LLM_RULES]
    if not targets:
        return "", ""
    if not llm.get_ml_llm_config().get("api_key"):
        return "ML LLM skipped: API key is not configured.", ""
    notes: list[str] = []
    last_stdout = ""
    for rule, filename in dict.fromkeys((str(f["rule"]), str(f["file"])) for f in targets):
        current = _current_findings(state)
        match = next((f for f in current if str(f["rule"]) == rule and str(f["file"]) == filename), None)
        if match is None:
            continue
        path = workspace / filename
        source = path.read_text(encoding="utf-8")
        try:
            fn = _enclosing_function(ast.parse(source), int(match["line"]))
        except SyntaxError:
            fn = None
        if fn is None:
            notes.append(f"{rule}: not inside a function, reported only")
            continue
        start = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
        func_source = "\n".join(source.splitlines()[start - 1:fn.end_lineno])
        user = f"Problem: {rule}. {match.get('message', '')}\nSuggestion: {match.get('suggestion', '')}\n\nFunction:\n{func_source}"
        before_counts = _rule_counts(current)
        try:
            answer = llm.complete_ml(LLM_SYSTEM, user)
            if isinstance(answer, dict):
                answer = json.dumps(answer)
            if not answer:
                raise ValueError(llm.ml_error_details() or "empty answer from the ML LLM")
            candidate = _build_candidate(source, fn, answer)
        except Exception as exc:
            rejected.append({"rule": rule, "reason": "LLM answer unusable", "error": str(exc)[:1500]})
            continue
        path.write_text(candidate, encoding="utf-8")
        after_counts = _rule_counts(_current_findings(state))
        fixed = after_counts.get(rule, 0) < before_counts.get(rule, 0)
        no_new = all(after_counts.get(r, 0) <= before_counts.get(r, 0) for r in after_counts)
        if not (fixed and no_new):
            path.write_text(source, encoding="utf-8")
            rejected.append({"rule": rule, "reason": "Finding still present or a new finding appeared", "error": ""})
            continue
        passed, error, stdout = _run_candidate(state, workspace)
        if not passed:
            path.write_text(source, encoding="utf-8")
            rejected.append({"rule": rule, "reason": "Candidate failed verification", "error": error[:1500]})
            continue
        applied.append({"rule": rule, "line": int(match["line"]), "method": "llm",
                        "description": f"LLM rewrote {fn.name}() to fix {rule}"})
        last_stdout = stdout or last_stdout
    return "; ".join(notes), last_stdout


def ml_fix(state: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic ML fixes, then LLM fixes, verifying each fix on its own."""
    if not state.get("enable_ml_fixes", True):
        return {"ml_fixes": {"enabled": False, "applied": [], "rejected": [], "report_only": [],
                             "note": "ML fixes disabled.", "behavior_changing": False}, "behavior_changing": False}
    workspace = Path(state.get("workspace_path", ""))
    pre = Path(tempfile.mkdtemp(prefix="agent1-ml-"))
    shutil.copytree(workspace, pre, dirs_exist_ok=True)
    _, _, stdout_before = _run_candidate(state, pre)
    stdout_before = stdout_before or str((state.get("baseline") or {}).get("stdout", ""))[:1500]
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    classification = state.get("problem_type") == "classification"
    findings = state.get("ml_review", {}).get("findings", [])

    # 1) deterministic fixes, one (file, rule) at a time
    for filename, rule in dict.fromkeys((str(f["file"]), str(f["rule"])) for f in findings):
        path = workspace / filename
        try:
            source = path.read_text(encoding="utf-8")
            updated, changes = _replace_calls(source, [{"rule": rule}], classification)
            if updated == source:
                continue
            path.write_text(updated, encoding="utf-8")
            passed, error, _ = _run_candidate(state, workspace)
            if passed:
                applied.extend(changes)
            else:
                path.write_text(source, encoding="utf-8")
                rejected.append({"rule": rule, "reason": "Candidate failed verification", "error": error[:1500]})
        except Exception as exc:
            rejected.append({"rule": rule, "reason": str(exc)[:200], "error": str(exc)[:1500]})

    # 2) LLM fixes for what is left
    note, _ = _llm_stage(state, workspace, applied, rejected)

    # 3) final combined run; revert everything only if it fails
    ok, error, final_stdout = _run_candidate(state, workspace)
    if not ok:
        shutil.copytree(pre, workspace, dirs_exist_ok=True)
        rejected.append({"rule": "ALL", "reason": "Final combined run failed; all ML fixes reverted", "error": error[:1500]})
        applied, final_stdout = [], stdout_before

    # 4) nothing disappears silently: leftover findings are report-only
    tried = {r["rule"] for r in rejected}
    report_only = [{"rule": f["rule"], "file": f["file"], "line": f["line"], "message": f.get("message", "")}
                   for f in _current_findings(state) if f["rule"] not in tried]
    return {
        "pre_ml_snapshot": str(pre),
        "behavior_changing": bool(applied),
        "ml_fixes": {"enabled": True, "applied": applied, "rejected": rejected, "report_only": report_only,
                     "note": note, "behavior_changing": bool(applied),
                     "stdout_before": stdout_before[:1500], "stdout_after": final_stdout[:1500], "ml_patch_path": ""},
    }