"""Infer the supervised learning target and feature groups."""

import re
from pathlib import Path
from typing import Any

import yaml

from ..config import (
    CLASSIFICATION_MAX_UNIQUE,
    CLASSIFICATION_UNIQUE_RATIO,
    ID_NAME_PARTS,
    IMBALANCE_RATIO,
    TARGET_COMMON_NAMES,
)
from ..state import Agent1State
from .target_from_code import target_from_code


def _manifest_target(workspace_path: str) -> str | None:
    """Read a target declared in mlops.yaml, if one exists."""
    try:
        document = yaml.safe_load(
            (Path(workspace_path) / "mlops.yaml").read_text(encoding="utf-8")
        )
        if isinstance(document, dict):
            data = document.get("data")
            if isinstance(data, dict) and isinstance(data.get("target"), str):
                return data["target"]
            if isinstance(document.get("target"), str):
                return document["target"]
    except Exception:
        pass
    return None


def _code_target(workspace_path: str) -> str | None:
    """Find target column references in Python source using narrow patterns."""
    constants: dict[str, str] = {}
    references: list[str] = []
    patterns = (
        r"(?:df|data|frame)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
        r"(?:df|data|frame)\s*\.pop\(\s*['\"]([^'\"]+)['\"]\s*\)",
    )
    variable_pattern = re.compile(
        r"(?:df|data|frame)\s*\[\s*([A-Za-z_]\w*)\s*\]"
    )
    for path in sorted(Path(workspace_path).rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        constants.update(
            {
                name: value
                for name, value in re.findall(
                    r"\b([A-Z][A-Z0-9_]*)\s*=\s*['\"]([^'\"]+)['\"]", source
                )
            }
        )
        for pattern in patterns:
            references.extend(re.findall(pattern, source))
        references.extend(
            constants[name]
            for name in variable_pattern.findall(source)
            if name in constants
        )
    return references[0] if references else None


def _future_llm_tie_breaker(profile: dict[str, Any]) -> None:
    """Placeholder for a future LLM tie-breaker when rules are ambiguous."""
    return None


def _is_id_like(name: str, unique_count: int, rows: int) -> bool:
    """Return whether a column looks like a row identifier."""
    lowered = name.lower()
    return rows > 0 and unique_count == rows and any(
        part in lowered for part in ID_NAME_PARTS
    )


TEXT_AVG_LENGTH = 30  # string columns with longer values on average are free text
TEXT_MIN_UNIQUE = 20  # free text has many distinct values; long category labels do not


def _text_columns(csv_path: str, names: list[str]) -> set[str]:
    """Return string columns that hold free text (long values) instead of categories."""
    if not csv_path or not names:
        return set()
    try:
        import pandas as pd

        frame = pd.read_csv(csv_path, usecols=lambda c: c in names, nrows=2000)
        return {
            name
            for name in names
            if name in frame
            and not pd.api.types.is_numeric_dtype(frame[name])
            and frame[name].dropna().astype(str).str.len().mean() > TEXT_AVG_LENGTH
            and frame[name].nunique() > TEXT_MIN_UNIQUE
        }
    except Exception:
        return set()


def detect_problem_type(state: Agent1State) -> dict[str, object]:
    """Detect target, problem type, subtype, and usable feature groups."""
    try:
        profile = dict(state.get("profile", {}))
        columns = profile.get("columns", {})
        if not isinstance(columns, dict) or not columns:
            raise ValueError("Data profile has no columns")
        workspace = state.get("workspace_path", "")
        manifest_target = _manifest_target(workspace)
        precise_target = target_from_code(workspace, columns.keys()) if workspace else None
        code_target = precise_target or (_code_target(workspace) if workspace else None)
        common_target = next(
            (name for name in columns if name.lower() in TARGET_COMMON_NAMES), None
        )
        target = manifest_target or code_target or common_target or next(
            iter(columns), None
        )
        source = (
            "manifest" if manifest_target
            else "code_precise" if precise_target
            else "code" if code_target
            else "common" if common_target
            else "fallback"
        )
        if manifest_target and code_target and manifest_target == code_target:
            source = "manifest_and_code"
        if target not in columns:
            raise ValueError(f"Detected target is not in profile: {target}")

        target_info = columns[target]
        rows = int(profile.get("rows", 0))
        unique_count = int(target_info.get("unique_count", 0))
        numeric = bool(target_info.get("numeric"))
        classification = not numeric or unique_count <= max(
            CLASSIFICATION_MAX_UNIQUE, int(rows * CLASSIFICATION_UNIQUE_RATIO)
        )
        problem_type = "classification" if classification else "regression"
        subtype = None
        imbalance = False
        if classification:
            subtype = "binary" if unique_count == 2 else "multiclass"
            if rows and unique_count > 1:
                counts = target_info.get("value_counts", {})
                if isinstance(counts, dict) and counts:
                    imbalance = min(counts.values()) / max(counts.values()) < IMBALANCE_RATIO
        numeric_features: list[str] = []
        categorical_features: list[str] = []
        for name, info in columns.items():
            if name == target or _is_id_like(name, int(info.get("unique_count", 0)), rows):
                continue
            (numeric_features if info.get("numeric") else categorical_features).append(name)
        text_set = _text_columns(state.get("csv_path", ""), categorical_features)
        text_features = [n for n in categorical_features if n in text_set]
        categorical_features = [n for n in categorical_features if n not in text_set]
        features: dict[str, list[str]] = {"numeric": numeric_features, "categorical": categorical_features}
        if text_features:
            features["text"] = text_features

        # Future LLM tie-breaker: call _future_llm_tie_breaker only for unresolved rules.
        confidence = {
            "manifest_and_code": 0.99,
            "manifest": 0.95,
            "code_precise": 0.95,
            "code": 0.90,
            "common": 0.75,
            "fallback": 0.55,
        }[source]
        profile["target_analysis"] = {
            "source": source,
            "class_imbalance": imbalance,
        }
        return {
            "profile": profile,
            "problem_type": problem_type,
            "subtype": subtype,
            "target": target,
            "features": features,
            "confidence": confidence,
        }
    except Exception as exc:
        return {
            "problem_type": None,
            "subtype": None,
            "target": None,
            "features": {"numeric": [], "categorical": []},
            "confidence": 0.0,
            "errors": [{"node": "detect_problem_type", "message": str(exc)}],
        }
