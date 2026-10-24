"""Drift d'une colonne : compare une fenêtre de production à sa référence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DriftConfig
from ..schemas import DatasetDrift, FeatureDriftResult
from . import stats
from .reference import ColumnReference


def _psi_level(psi: float, cfg: DriftConfig) -> str:
    if psi >= cfg.psi_major:
        return "major"
    if psi >= cfg.psi_moderate:
        return "moderate"
    return "none"


def column_drift(ref: ColumnReference, current: pd.Series, cfg: DriftConfig) -> FeatureDriftResult:
    if ref.kind == "numeric":
        cur = stats.clean_numeric(current)
        psi = stats.psi_numeric(ref.values, cur, ref.edges)
        if len(cur) > cfg.large_sample:
            method, score_type, threshold = "wasserstein", "distance", cfg.wasserstein
            score = stats.wasserstein_normed(ref.values, cur)
            drifted = score > threshold
        else:
            method, score_type, threshold = "ks", "p_value", cfg.p_value
            _, score = stats.ks_test(ref.values, cur)
            drifted = score < threshold
        current_summary = ({"mean": float(cur.mean()), "std": float(cur.std()), "median": float(np.median(cur))}
                           if len(cur) else {})
    else:
        cur_str = stats.as_category_strings(current)
        # On compare sur l'union des catégories : une catégorie nouvelle compte dans le drift.
        new = sorted(set(cur_str.unique()) - set(ref.categories))
        categories = ref.categories + new
        ref_counts = np.concatenate([ref.counts, np.zeros(len(new))])
        cur_counts = stats.category_counts(cur_str, categories)
        p_ref = ref_counts / ref_counts.sum() if ref_counts.sum() else ref_counts
        p_cur = cur_counts / cur_counts.sum() if cur_counts.sum() else cur_counts
        psi = stats.psi_from_proportions(p_ref, p_cur) if cur_counts.sum() else 0.0
        if cur_counts.sum() > cfg.large_sample:
            method, score_type, threshold = "jensen_shannon", "distance", cfg.jensen_shannon
            score = stats.jensen_shannon(p_ref, p_cur)
            drifted = score > threshold
        else:
            method, score_type, threshold = "chi2", "p_value", cfg.p_value
            score = stats.chi2_pvalue(ref_counts, cur_counts)
            drifted = score < threshold
        top = np.argsort(-cur_counts)[:3]
        current_summary = {categories[i]: float(p_cur[i]) for i in top} if cur_counts.sum() else {}

    return FeatureDriftResult(
        feature=ref.name, kind=ref.kind, method=method, score_type=score_type,
        score=round(float(score), 6), threshold=threshold, drifted=bool(drifted),
        psi=round(float(psi), 6), psi_level=_psi_level(psi, cfg),
        reference_summary=ref.summary(), current_summary=current_summary,
    )


def dataset_drift(results: list[FeatureDriftResult], cfg: DriftConfig) -> DatasetDrift:
    drifted = [r.feature for r in results if r.drifted]
    n = len(results)
    share = len(drifted) / n if n else 0.0
    return DatasetDrift(n_features=n, n_drifted=len(drifted), share_drifted=round(share, 4),
                        drifted=share >= cfg.dataset_share, drifted_features=drifted)
