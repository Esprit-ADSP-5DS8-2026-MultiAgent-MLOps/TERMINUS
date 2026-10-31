"""Find the target column by reading how the project's code uses it."""

import ast
from pathlib import Path
from typing import Any, Iterable


def _constant(node: ast.AST, assigns: dict[str, ast.AST]) -> ast.AST:
    """Replace a name such as TARGET by the value it was assigned (TARGET = "target")."""
    if isinstance(node, ast.Name) and isinstance(assigns.get(node.id), ast.Constant):
        return assigns[node.id]
    return node


def _column_from_expr(node: ast.AST, assigns: dict[str, ast.AST], columns: set[str], depth: int = 0) -> str | None:
    """Resolve df["col"], df[TARGET], df.pop("col") or a name assigned to one of them."""
    if depth > 5:
        return None
    if isinstance(node, ast.Name) and node.id in assigns:
        return _column_from_expr(assigns[node.id], assigns, columns, depth + 1)
    if isinstance(node, ast.Subscript):
        key = _constant(node.slice, assigns)
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in columns:
            return key.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop":
        arg = _constant(node.args[0], assigns) if node.args else None
        if isinstance(arg, ast.Constant) and arg.value in columns:
            return str(arg.value)
    return None


def _assignments(tree: ast.AST) -> dict[str, ast.AST]:
    """name -> assigned value, including tuple unpacking: X, y = df.drop(...), df[TARGET]."""
    assigns: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            assigns[target.id] = node.value
        elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                and len(target.elts) == len(node.value.elts):
            for name, value in zip(target.elts, node.value.elts):
                if isinstance(name, ast.Name):
                    assigns[name.id] = value
    return assigns


def target_from_code(workspace: str | Path, columns: Iterable[Any]) -> str | None:
    """Return the target column used by train_test_split(...) or .fit(X, y), or None."""
    known = {str(c) for c in columns}
    for path in sorted(Path(workspace).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        assigns = _assignments(tree)
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
