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


def _labelled(tmp_path, labels=("bad", "good"), name="train.csv"):
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["churn"] = [labels[int(v)] for v in y]
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return ResolvedPaths(tmp_path, csv, None)


def test_label_classes_travel_with_the_model(tmp_path, monkeypatch):
    """L'encodage des étiquettes est enregistré DANS le modèle MLflow : l'API rend « bad »/« good »."""
    from mlflow_agent.executor import label_classes_of

    r = run(tmp_path, monkeypatch, report(0.6), _labelled(tmp_path))
    assert r.status == "success", r.errors
    assert r.label_classes == ["bad", "good"]
    assert label_classes_of(r.model_uri) == ["bad", "good"]


def test_champion_is_scored_on_the_candidate_test_set(tmp_path, monkeypatch):
    """Même données, même graine : le champion rejugé sur le jeu de test du candidat obtient le même score."""
    from mlflow.tracking import MlflowClient

    paths = _labelled(tmp_path)
    first = run(tmp_path, monkeypatch, report(0.6), paths)
    MlflowClient().set_registered_model_alias("test_exp-model", "champion", first.model_version)
    second = run_experiment(report(0.6), PLAN, paths, tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                            experiment_name="test_exp", champion_uri="models:/test_exp-model@champion")
    assert second.champion_test_score == pytest.approx(second.test_score)
    assert second.champion_test_score == pytest.approx(first.test_score)


def test_champion_trained_on_other_classes_is_not_compared_blindly(tmp_path, monkeypatch):
    """Le nouveau jeu contient une étiquette que le champion n'a jamais vue : pas de score inventé."""
    from mlflow.tracking import MlflowClient

    first = run(tmp_path, monkeypatch, report(0.6), _labelled(tmp_path, ("bad", "good")))
    MlflowClient().set_registered_model_alias("test_exp-model", "champion", first.model_version)
    other = _labelled(tmp_path, ("bad", "risky"), name="new.csv")
    second = run_experiment(report(0.6), PLAN, other, tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                            experiment_name="test_exp", champion_uri="models:/test_exp-model@champion")
    assert second.champion_test_score is None, (second.status, second.errors)
    assert any("ne connaît pas toutes les classes" in w for w in second.warnings), (second.status, second.errors, second.warnings)


def test_overfit_is_measured_on_the_gain_over_the_naive_model():
    """Même règle pour une erreur (rmse) et un score (roc_auc) : la part du gain perdue en validation."""
    from mlflow_agent.executor import MAX_OVERFIT_SHARE, overfit_share, quality_gate

    # cas réel (house_prices réentraîné) : rmse train 10, validation 31,5, modèle naïf 100,8 -> il généralise
    assert overfit_share("rmse", 10.0, 31.46, 100.77) < MAX_OVERFIT_SHARE
    assert quality_gate("rmse", 40, 10.0, 31.46, 34.92, 100.77, 100.77).passed
    # l'ancienne règle, (val - train) / train = 2,1, l'aurait rejeté ; un vrai sur-apprentissage reste rejeté
    # validation -> test, même échelle : 28,5 -> 36,6 sur un modèle naïf à 101 reste cohérent (réel, 2e passage)
    assert quality_gate("rmse", 40, 10.0, 28.54, 36.61, 101.10, 101.10).checks["test_consistent_with_validation"]
    assert not quality_gate("rmse", 90, 10.0, 28.54, 80.0, 101.10, 101.10).checks["test_consistent_with_validation"]
    assert overfit_share("roc_auc", 1.0, 0.6, 0.5) > MAX_OVERFIT_SHARE
    assert not quality_gate("roc_auc", 0.55, 1.0, 0.6, 0.6, 0.5, 0.5).checks["overfit_gap_ok"]


def test_an_overfitting_best_model_gives_way_to_the_next(tmp_path, monkeypatch):
    """Le meilleur en validation croisée sur-apprend : le suivant, qui généralise, est retenu."""
    from mlflow_agent import executor

    plan = ExperimentPlan(models=["logistic_regression", "random_forest"], primary_metric="roc_auc",
                          search_method="optuna", max_trials=2, rationale="test")
    real = executor._generalization

    def fake(name, *a, **k):
        train, val, base = real(name, *a, **k)
        return (1.0, 0.55, 0.5) if name == "random_forest" else (train, val, base)

    monkeypatch.setattr(executor, "_generalization", fake)
    monkeypatch.setattr(executor, "optimize_model",
                        lambda name, *a, **k: ({}, 0.99 if name == "random_forest" else 0.9, 1))
    monkeypatch.chdir(tmp_path)
    r = run_experiment(report(0.6), plan, env(tmp_path), tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
                       experiment_name="test_exp")
    assert r.best_model_name == "logistic_regression", (r.errors, r.warnings)
    assert any("random_forest écarté" in w for w in r.warnings)


def test_float_labels_from_production_stay_integer_classes(tmp_path, monkeypatch):
    """Réentraînement : la gateway rend les labels en 0.0 / 1.0 ; les classes enregistrées restent 0 et 1
    (sinon l'API servait « 1 » en texte et ses tests échouaient)."""
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["churn"] = y.astype(float)
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    r = run(tmp_path, monkeypatch, report(0.6), ResolvedPaths(tmp_path, csv, None))
    assert r.status == "success", r.errors
    assert r.label_classes == [0, 1] and all(type(c) is int for c in r.label_classes)
