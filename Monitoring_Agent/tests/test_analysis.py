"""Tests du moteur d'analyse sur des données synthétiques (aucun fichier requis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring_agent.analysis import Analyzer, make_windows, stats
from monitoring_agent.analysis.performance import CBPE, realized_metrics
from monitoring_agent.config import AnalyzerConfig
from monitoring_agent.schemas import ColumnMapping

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------


def test_same_distribution_is_not_drift():
    a, b = RNG.normal(0, 1, 5000), RNG.normal(0, 1, 5000)
    assert stats.wasserstein_normed(a, b) < 0.1
    assert stats.psi_numeric(a, b, stats.quantile_edges(a)) < 0.1


def test_shifted_distribution_is_drift():
    a, b = RNG.normal(0, 1, 5000), RNG.normal(0.5, 1, 5000)
    assert stats.wasserstein_normed(a, b) > 0.1
    assert stats.psi_numeric(a, b, stats.quantile_edges(a)) > 0.1


def test_categorical_distances():
    p = np.array([0.5, 0.3, 0.2])
    assert stats.jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-9)
    assert stats.jensen_shannon(p, np.array([0.1, 0.3, 0.6])) > 0.1
    assert stats.chi2_pvalue(np.array([500, 300, 200]), np.array([50, 30, 20])) > 0.5


def test_booleans_and_strings_are_one_category():
    assert stats.as_category_strings(pd.Series([True, False])).tolist() == ["True", "False"]


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def _calibrated_sample(n: int, a: float = 2.0, b: float = 5.0):
    """Probabilités calibrées par construction : y ~ Bernoulli(p)."""
    p = RNG.beta(a, b, n)
    y = (RNG.random(n) < p).astype(int)
    return p, y, (p >= 0.5).astype(int)


def test_cbpe_estimates_without_labels():
    p_ref, y_ref, _ = _calibrated_sample(20000)
    cbpe = CBPE().fit(p_ref, y_ref)
    # production DÉPLACÉE (plus de cas incertains) : la performance réelle baisse
    p, y, yp = _calibrated_sample(20000, a=4, b=4)
    est = cbpe.estimate(p, yp)
    real = realized_metrics("classification", y, yp, p)
    for m in ("accuracy", "f1", "roc_auc"):
        assert est[m] == pytest.approx(real[m], abs=0.02), m


# ---------------------------------------------------------------------------
# Analyzer de bout en bout
# ---------------------------------------------------------------------------


def _frame(n: int, shift: float = 0.0, missing: float = 0.0, new_cat: bool = False) -> pd.DataFrame:
    x = RNG.normal(shift, 1, n)
    cat = RNG.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]).astype(object)
    if new_cat:
        cat[: n // 10] = "z"
    logit = 1.5 * x + (cat == "a") * 0.5
    p = 1 / (1 + np.exp(-logit))
    y = (RNG.random(n) < p).astype(int)
    df = pd.DataFrame({"x": x, "cat": cat, "proba": p, "pred": (p >= 0.5).astype(int), "y": y})
    if missing:
        df.loc[df.sample(frac=missing, random_state=1).index, "x"] = np.nan
    return df


MAPPING = ColumnMapping(task="classification", features=["x", "cat"], target="y",
                        prediction="pred", prediction_proba="proba", primary_metric="roc_auc")


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer(_frame(20000), MAPPING, AnalyzerConfig(window_size=2000))


def test_quiet_window_does_not_wake_the_agent(analyzer):
    w = analyzer.analyze_window(_frame(2000))
    assert w.dataset_drift.n_drifted == 0
    assert not w.trigger_agent


def test_drift_is_detected(analyzer):
    w = analyzer.analyze_window(_frame(2000, shift=1.0))
    assert "x" in w.dataset_drift.drifted_features
    assert w.trigger_agent


def test_quality_issues_are_detected(analyzer):
    w = analyzer.analyze_window(_frame(2000, missing=0.3, new_cat=True))
    kinds = {(q.feature, q.kind) for q in w.data_quality}
    assert ("x", "missing_values") in kinds
    assert ("cat", "unseen_category") in kinds
    assert any(s.kind == "data_quality" and s.severity == "critical" for s in w.signals)


def test_missing_column_is_critical(analyzer):
    w = analyzer.analyze_window(_frame(2000).drop(columns=["cat"]))
    assert any(q.kind == "missing_column" for q in w.data_quality)
    assert w.severity == "critical"


def test_estimated_performance_without_labels(analyzer):
    w = analyzer.analyze_window(_frame(2000).assign(y=np.nan))
    assert w.performance.labels_available is False
    assert w.performance.realized is None
    assert w.performance.estimated["roc_auc"] > 0.7


def test_windows():
    assert [len(w) for w in make_windows(pd.DataFrame({"a": range(10500)}), 5000)] == [5000, 5500]
