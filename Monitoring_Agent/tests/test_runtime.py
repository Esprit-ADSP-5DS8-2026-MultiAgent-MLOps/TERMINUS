"""Gateway, moniteur et génération des configs, avec une FAUSSE API de modèle (aucun réseau)."""

from __future__ import annotations

import json
import math

import httpx
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitoring_agent.provisioning.grafana import build_dashboard
from monitoring_agent.provisioning.prometheus import build_rules, check_rules, write_rules
from monitoring_agent.runtime.gateway import create_app
from monitoring_agent.runtime.target import MonitoringTarget
from monitoring_agent.schemas import ColumnMapping

RNG = np.random.default_rng(1)


def fake_model() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(row: dict):
        if row.get("x") is None:
            from fastapi import HTTPException
            raise HTTPException(422, "x manquant")
        p = 1 / (1 + math.exp(-1.5 * row["x"]))
        return {"prediction": int(p >= 0.5), "proba": p}

    return app


def rows(n: int, shift: float = 0.0) -> pd.DataFrame:
    x = RNG.normal(shift, 1, n)
    cat = RNG.choice(["a", "b"], n)
    p = 1 / (1 + np.exp(-1.5 * x))
    return pd.DataFrame({"x": x, "cat": cat, "target": (RNG.random(n) < p).astype(int),
                         "y_pred": (p >= 0.5).astype(int), "y_pred_proba": p})


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_RUNS_DIR", str(tmp_path))
    t = MonitoringTarget(
        project="demo", upstream_url="http://model", window_size=100,
        mapping=ColumnMapping(task="classification", features=["x", "cat"], target="target", prediction="y_pred",
                              prediction_proba="y_pred_proba", timestamp="timestamp", id="request_id",
                              primary_metric="roc_auc", min_threshold=0.7))
    t.dir.mkdir(parents=True)
    rows(2000).to_parquet(t.reference_path)
    t.save()
    return t


@pytest.fixture
def client(target):
    app = create_app(target, transport=httpx.ASGITransport(app=fake_model()), run_monitor=False)
    with TestClient(app) as c:
        yield c


def send(client, frame: pd.DataFrame) -> list[str]:
    ids = []
    for rec in frame[["x", "cat"]].to_dict(orient="records"):
        r = client.post("/predict", json=rec)
        assert r.status_code == 200
        ids.append(r.headers["x-request-id"])
    return ids


def test_proxy_is_transparent(client):
    r = client.post("/predict", json={"x": 2.0, "cat": "a"}, headers={"X-Request-ID": "loan-42"})
    assert r.status_code == 200
    assert r.json()["prediction"] == 1 and 0.9 < r.json()["proba"] < 1
    assert r.headers["x-request-id"] == "loan-42"


def test_upstream_errors_are_forwarded_and_counted(client):
    r = client.post("/predict", json={"cat": "a"})
    assert r.status_code == 422
    text = client.get("/metrics").text
    assert 'mlmon_http_requests_total{project="demo",route="/predict",status="422"} 1.0' in text


def test_windows_drift_and_late_labels(client):
    app = client.app
    calm, shifted = rows(100), rows(100, shift=1.5)
    ids = send(client, calm) + send(client, shifted)
    new = app.state.monitor.step()
    assert [w.window.index for w in new] == [0, 1]
    assert not new[0].trigger_agent
    assert "x" in new[1].dataset_drift.drifted_features and new[1].trigger_agent
    assert new[1].performance.labels_available is False        # pas encore de vérité terrain
    assert new[1].performance.estimated is not None

    labels = calm["target"].tolist() + shifted["target"].tolist()
    r = client.post("/feedback", json={"items": [{"request_id": i, "label": y} for i, y in zip(ids, labels)]})
    assert r.json() == {"received": 200, "matched": 200}
    app.state.monitor.step()                                   # les labels arrivent : perf mesurée
    reports = client.get("/monitoring/windows").json()
    assert all(w["performance"]["labels_available"] for w in reports)

    metrics = client.get("/metrics").text
    assert 'mlmon_feature_drifted{feature="x",project="demo"} 1.0' in metrics
    assert "mlmon_performance_realized" in metrics and "mlmon_agent_triggers_total" in metrics


def test_alertmanager_webhook(client):
    payload = {"alerts": [{"status": "firing", "labels": {"alertname": "MLDatasetDrift", "severity": "warning"}}]}
    assert client.post("/alerts", json=payload).json() == {"stored": 1}
    assert client.get("/monitoring/alerts").json()[0]["alertname"] == "MLDatasetDrift"


def test_rules_are_project_specific(target, tmp_path):
    rules = build_rules(target)
    names = [r["alert"] for g in rules["groups"] for r in g["rules"]]
    assert {"MLDatasetDrift", "MLPerformanceDropEstimated", "MLBelowBusinessThreshold"} <= set(names)
    expr = next(r["expr"] for g in rules["groups"] for r in g["rules"] if r["alert"] == "MLPerformanceDropEstimated")
    assert 'metric="roc_auc"' in expr and " < on(project, metric) " in expr     # roc_auc : plus haut = mieux
    path = write_rules(target, tmp_path)
    assert yaml.safe_load(path.read_text())["groups"]
    ok, out = check_rules(path)
    assert ok is not False, out                                 # None si promtool absent


def test_dashboard_layout(target):
    d = build_dashboard(target, bound=0.9, reference=0.95)
    ids = [p["id"] for p in d["panels"]]
    assert len(ids) == len(set(ids))
    for p in d["panels"]:
        g = p["gridPos"]
        assert g["x"] + g["w"] <= 24
    assert json.dumps(d)                                        # sérialisable tel quel
