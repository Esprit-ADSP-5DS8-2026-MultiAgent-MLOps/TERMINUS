import copy

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("optuna")
pytest.importorskip("mlflow")

from sklearn.datasets import make_classification  # noqa: E402

from mlflow_agent.executor import run_experiment  # noqa: E402
from mlflow_agent.plans import ExperimentPlan  # noqa: E402
from mlflow_agent.schemas import AnalysisReport, ResolvedPaths  # noqa: E402
from tests.test_mlflow_agent_input import SAMPLE  # noqa: E402

PLAN = ExperimentPlan(models=["logistic_regression"], primary_metric="roc_auc",
                      search_method="optuna", max_trials=2, rationale="test")


def env(tmp_path, with_target=True):
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    if with_target:
        df["churn"] = y
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    return ResolvedPaths(tmp_path, csv, None)


def report(min_metric):
    d = copy.deepcopy(SAMPLE)
    d["task"]["primary_metric"] = "roc_auc"
    d["task"]["min_primary_metric"] = min_metric
    return AnalysisReport.model_validate(d)


def run(tmp_path, monkeypatch, rep, paths):
    monkeypatch.chdir(tmp_path)  # keep mlruns/ artifacts inside tmp
    return run_experiment(rep, PLAN, paths, tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                          experiment_name="test_exp")


def test_success_registers_model(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, report(0.6), env(tmp_path))
    assert r.status == "success", r.errors
    assert r.quality_gate.passed and r.model_uri.startswith("models:/")
    assert r.reproducibility["test_set_scored_once"] is True


def test_impossible_threshold_means_no_eligible_model(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, report(1.5), env(tmp_path))
    assert r.status == "no_eligible_model"
    assert r.model_uri is None and not r.quality_gate.checks["meets_threshold"]


def test_missing_target_column_fails_cleanly(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, report(0.6), env(tmp_path, with_target=False))
    assert r.status == "failed" and r.errors
