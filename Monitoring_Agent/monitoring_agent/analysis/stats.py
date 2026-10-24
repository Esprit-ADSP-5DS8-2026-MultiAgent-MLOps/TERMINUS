"""
Mesures de drift : fonctions pures, sans état, testables une par une.

Deux familles :
- TESTS statistiques (KS, chi²) : « les deux échantillons viennent-ils de la même loi ? ».
  Sur 5 000 lignes, la p-value devient quasi nulle pour des écarts sans importance.
- DISTANCES (Wasserstein normalisée, Jensen-Shannon, PSI) : mesurent l'AMPLEUR de l'écart.
  En production, c'est ce qui compte.

Règle (celle d'Evidently) : petits échantillons -> tests ; gros échantillons -> distances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

EPS = 1e-6


def clean_numeric(values) -> np.ndarray:
    """Valeurs numériques sans NaN (les manquants sont traités par les contrôles qualité)."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# Numérique
# ---------------------------------------------------------------------------


def quantile_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """Bornes des classes pour le PSI : quantiles de la RÉFÉRENCE (classes de même effectif).
    Les bornes extrêmes sont ouvertes (-inf, +inf) : une valeur nouvelle tombe dans une classe."""
    if len(reference) == 0:
        return np.array([-np.inf, np.inf])
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1))[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    if len(values) == 0:
        return np.zeros(n_bins)
    idx = np.searchsorted(edges[1:-1], values, side="right")
    return np.bincount(idx, minlength=n_bins) / len(values)


def psi_from_proportions(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """Population Stability Index : somme de (cur - ref) * ln(cur / ref)."""
    p_ref = np.clip(np.asarray(p_ref, dtype=float), EPS, None)
    p_cur = np.clip(np.asarray(p_cur, dtype=float), EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_numeric(reference: np.ndarray, current: np.ndarray, edges: np.ndarray) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    return psi_from_proportions(bin_proportions(reference, edges), bin_proportions(current, edges))


def wasserstein_normed(reference: np.ndarray, current: np.ndarray) -> float:
    """Distance de Wasserstein divisée par l'écart-type de la référence.
    0.1 = la distribution s'est déplacée d'environ 0.1 écart-type en moyenne."""
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    dist = stats.wasserstein_distance(reference, current)
    std = float(np.std(reference))
    if std == 0:
        return 0.0 if dist == 0 else float("inf")
    return float(dist / std)


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov : (statistique D, p-value)."""
    if len(reference) == 0 or len(current) == 0:
        return 0.0, 1.0
    res = stats.ks_2samp(reference, current)
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# Catégoriel
# ---------------------------------------------------------------------------


def as_category_strings(values) -> pd.Series:
    """Catégories normalisées en texte (True, 'True' et 1.0 ne doivent pas devenir 3 modalités)."""
    s = pd.Series(values)
    s = s[s.notna()]
    if pd.api.types.is_bool_dtype(s):
        return s.map({True: "True", False: "False"})
    return s.astype(str)


def category_counts(values, categories: list[str]) -> np.ndarray:
    counts = as_category_strings(values).value_counts()
    return np.array([counts.get(c, 0) for c in categories], dtype=float)


def jensen_shannon(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """Distance de Jensen-Shannon en base 2 : 0 = identiques, 1 = totalement disjointes."""
    if p_ref.sum() == 0 or p_cur.sum() == 0:
        return 0.0
    return float(jensenshannon(p_ref, p_cur, base=2))


def chi2_pvalue(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    table = np.vstack([ref_counts, cur_counts])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return 1.0
    return float(stats.chi2_contingency(table)[1])
