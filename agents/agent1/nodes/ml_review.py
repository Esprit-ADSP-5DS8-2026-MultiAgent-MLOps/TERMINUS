"""Report-only ML practice rules."""

import ast
from pathlib import Path
from typing import Any

import pandas as pd


def _finding(file: str, line: int, rule: str, severity: str, message: str, suggestion: str) -> dict[str, str | int]:
    """Create a normalized ML review finding."""
    return {"file": file, "line": line, "rule": rule, "severity": severity, "message": message, "suggestion": suggestion}


def run_rules(source: str, context: dict[str, Any]) -> list[dict[str, str | int]]:
    """Run reusable AST ML rules against one source file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings: list[dict[str, str | int]] = []
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "" for node in calls}
    split = next((node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "train_test_split"), None)
    split_line = split.lineno if split else 10**9
    leakage = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"} and node.lineno < split_line]
    for node in leakage:
        findings.append(_finding(context.get("file", ""), node.lineno, "DATA_LEAKAGE", "high", "Preprocessing is fitted before the train/test split.", "Fit preprocessing only on training data, preferably in an sklearn Pipeline."))
    random_calls = {"train_test_split", "LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier", "SVC", "LinearSVC", "SGDClassifier"}
    for node in calls:
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name in random_calls and not any(keyword.arg == "random_state" for keyword in node.keywords):
            findings.append(_finding(context.get("file", ""), node.lineno, "NO_RANDOM_STATE", "medium", f"{name} has no fixed random_state.", "Set random_state=42 for reproducibility."))
    imbalanced = bool(context.get("imbalanced"))
    # stratify n'a de sens que pour des classes : pas de faux problème « high » sur une régression
    if split and context.get("problem_type") != "regression" and not any(keyword.arg == "stratify" for keyword in split.keywords):
        findings.append(_finding(context.get("file", ""), split.lineno, "NO_STRATIFY", "high" if imbalanced else "medium", "Classification split does not preserve class proportions.", "Pass stratify=y to train_test_split."))
    classification = context.get("problem_type") == "classification"
    source_lower = source.lower()
    handled = "class_weight" in source_lower or any(word in source_lower for word in ("smote", "oversampling", "undersampling", "resample"))
    accuracy_only = "accuracy_score" in source_lower and not any(metric in source_lower for metric in ("f1_score", "roc_auc_score", "precision_score", "recall_score"))
    if classification and imbalanced and not handled and accuracy_only:
        findings.append(_finding(context.get("file", ""), 1, "IMBALANCE_NOT_HANDLED", "high", "An imbalanced classification problem uses accuracy alone without imbalance handling.", "Use class_weight=balanced and report balanced metrics."))
    if classification and "accuracy" in source_lower and any(metric in source_lower for metric in ("price", "mse", "mean_squared_error")):
        findings.append(_finding(context.get("file", ""), 1, "WRONG_METRIC", "high", "The selected metric does not match the detected problem type.", "Use classification metrics such as F1 or ROC AUC."))
    if not any(name in names for name in {"cross_val_score", "cross_validate", "KFold", "StratifiedKFold"}):
        findings.append(_finding(context.get("file", ""), 1, "NO_CROSS_VALIDATION", "low", "No cross-validation is present.", "Use cross-validation to estimate generalization."))
    return findings


def ml_review(state: dict[str, Any]) -> dict[str, Any]:
    """Review every parseable workspace Python file without editing it."""
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    context = {"problem_type": state.get("problem_type"), "subtype": state.get("subtype"), "target": state.get("target")}
    context["imbalanced"] = state.get("profile", {}).get("target_analysis", {}).get("class_imbalance", False)
    if not context["imbalanced"]:
        try:
            frame = pd.read_csv(state.get("csv_path", ""))
            target = state.get("target") or (str(frame.columns[-1]) if len(frame.columns) else None)
            if target and target in frame:
                counts = frame[target].value_counts()
                context["imbalanced"] = len(counts) > 1 and counts.min() / counts.max() < 0.2
        except Exception as exc:
            errors.append({"node": "ml_review", "message": str(exc)})
    workspace = Path(state.get("workspace_path", ""))
    for path in sorted(workspace.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            findings.extend(run_rules(source, {**context, "file": str(path.relative_to(workspace))}))
        except (OSError, UnicodeDecodeError) as exc:
            errors.append({"node": "ml_review", "message": str(exc), "file": str(path)})
    return {"ml_review": {"high": sum(item["severity"] == "high" for item in findings), "medium": sum(item["severity"] == "medium" for item in findings), "low": sum(item["severity"] == "low" for item in findings), "findings": findings}, "tool_errors": [*state.get("tool_errors", []), *errors]}
