"""
Qualité des données d'une fenêtre.

Pourquoi c'est séparé du drift : un drift CAUSÉ par un bug de données (colonne vide,
format changé, nouvelle catégorie mal encodée) ne se corrige pas en réentraînant.
Réentraîner sur des données cassées, c'est apprendre le bug. L'agent doit donc savoir
distinguer « le monde a changé » de « le pipeline de données est cassé ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import QualityConfig
from ..schemas import QualityIssue
from . import stats
from .reference import ColumnReference


def column_quality(ref: ColumnReference, current: pd.DataFrame, cfg: QualityConfig) -> list[QualityIssue]:
    name = ref.name
    if name not in current.columns:
        return [QualityIssue(feature=name, kind="missing_column", value=1.0, severity="critical",
                             detail="colonne absente des données reçues")]
    s = current[name]
    n = len(s)
    if n == 0:
        return []
    issues: list[QualityIssue] = []

    missing = float(s.isna().mean())
    if missing - ref.missing_rate > cfg.missing_increase:
        sev = "critical" if missing >= cfg.missing_critical else "warning"
        issues.append(QualityIssue(
            feature=name, kind="missing_values", value=round(missing, 4),
            reference_value=round(ref.missing_rate, 4), severity=sev,
            detail=f"{missing:.1%} de valeurs manquantes (référence : {ref.missing_rate:.1%})"))

    present = s[s.notna()]
    if len(present) == 0:
        return issues

    if ref.kind == "numeric":
        coerced = pd.to_numeric(present, errors="coerce")
        bad = float(coerced.isna().mean())
        if bad > cfg.type_mismatch_rate:
            examples = present[coerced.isna()].astype(str).unique()[:3].tolist()
            issues.append(QualityIssue(
                feature=name, kind="type_mismatch", value=round(bad, 4), severity="critical",
                detail=f"valeurs non numériques, ex. {examples}"))
        vals = coerced.dropna().to_numpy(dtype=float)
        if len(vals) and ref.min is not None:
            out = float(np.mean((vals < ref.min) | (vals > ref.max)))
            if out > cfg.out_of_range_rate:
                # Un peu hors bornes : le monde bouge (info). Massivement hors bornes : bug d'unité / de format.
                issues.append(QualityIssue(
                    feature=name, kind="out_of_range", value=round(out, 4),
                    severity="warning" if out >= cfg.out_of_range_warning else "info",
                    detail=f"{out:.1%} hors de l'intervalle vu à l'entraînement [{ref.min:g}, {ref.max:g}]"))
    else:
        cats = stats.as_category_strings(present)
        unseen_mask = ~cats.isin(ref.categories)
        unseen = float(unseen_mask.mean())
        if unseen > cfg.unseen_rate:
            sev = "critical" if unseen >= cfg.unseen_critical else "warning"
            examples = cats[unseen_mask].value_counts().index[:3].tolist()
            issues.append(QualityIssue(
                feature=name, kind="unseen_category", value=round(unseen, 4), reference_value=0.0,
                severity=sev, detail=f"catégories jamais vues à l'entraînement, ex. {examples}"))
    return issues
