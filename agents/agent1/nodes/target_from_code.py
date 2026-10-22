"""Find the target column by reading how the project's code uses it."""

import ast
from pathlib import Path
from typing import Any, Iterable


def _column_from_expr(node: ast.AST, assigns: dict[str, ast.AST], columns: set[str]) -> str | None:
    """Resolve df["col"], df.pop("col") or a name assigned to one of them."""
    if isinstance(node, ast.Name) and node.id in assigns:
        return _column_from_expr(assigns[node.id], {}, columns)
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in columns:
            return key.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop":
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in columns:
            return str(node.args[0].value)
    return None


def target_from_code(workspace: str | Path, columns: Iterable[Any]) -> str | None:
    """Return the target column used by train_test_split(...) or .fit(X, y), or None."""
    known = {str(c) for c in columns}
    for path in sorted(Path(workspace).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        assigns: dict[str, ast.AST] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assigns[node.targets[0].id] = node.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            candidates: list[ast.AST] = []
            if name == "train_test_split":
                candidates += [k.value for k in node.keywords if k.arg == "stratify"]
                candidates += node.args[1:2]
            elif name == "fit":
                candidates += node.args[1:2]
            for expr in candidates:
                column = _column_from_expr(expr, assigns, known)
                if column:
                    return column
    return None
