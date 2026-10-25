"""Tests sur les vrais jeux NannyML (ignorés si data/ n'est pas présent)."""

from __future__ import annotations

import pytest

from monitoring_agent.analysis import Analyzer
from monitoring_agent.datasets import find_file, load


def _available(name: str) -> bool:
    try:
        find_file(name)
        return True
    except FileNotFoundError:
        return False


needs_car_loan = pytest.mark.skipif(not _available("synthetic_car_loan_reference.csv"),
                                    reason="données NannyML absentes")


@needs_car_loan
def test_car_loan_drift_starts_at_window_5():
    ref, prod, spec = load("car_loan")
    report = Analyzer(ref, spec.mapping).analyze(prod, "car_loan")
    assert report.summary["first_drift_window"] == 5
    assert all(not w.trigger_agent for w in report.windows[:5])
    # CBPE suit la vraie ROC AUC à 0.01 près, sans utiliser les labels
    for w in report.windows:
        assert abs(w.performance.estimated["roc_auc"] - w.performance.realized["roc_auc"]) < 0.01


@needs_car_loan
def test_label_delay_keeps_the_estimate():
    ref, prod, spec = load("car_loan", label_delay_windows=2)
    report = Analyzer(ref, spec.mapping).analyze(prod)
    last = report.windows[-1].performance
    assert last.labels_available is False and last.estimated is not None


@needs_car_loan
def test_evaluation_rules_baseline(tmp_path):
    from monitoring_agent.evaluation import evaluate
    s = evaluate("rules", per_dataset=1, out_dir=tmp_path, progress=lambda m: None)
    assert s["cases"] == 3 and s["agent_unsafe"] == 0 and s["calm_false_wakeups"] == 0
