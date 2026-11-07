"""Boucle Agent 4 -> orchestrateur : la demande de la gateway est remise une fois, avec ses données ;
les nouvelles données couvrent l'épisode en cours, drift des entrées OU baisse de performance."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from monitoring_agent.pipeline import _gateway_request
from monitoring_agent.runtime.supervisor import Supervisor


def test_gateway_request_is_handed_over_once(tmp_path):
    actions = tmp_path / "target" / "actions"
    actions.mkdir(parents=True)
    data = actions / "new_data_D1.parquet"
    pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_parquet(data)
    (actions / "retraining_request.json").write_text(json.dumps({
        "schema_version": "0.2", "model_uri": "models:/p/1", "reason": "performance_drop",
        "new_data_path": str(data), "diagnosis": "[concept_drift] roc_auc 0.64"}))
    target, wd = SimpleNamespace(dir=tmp_path / "target"), tmp_path / "wd"
    wd.mkdir()
    outputs, warnings = {}, []
    assert _gateway_request(target, wd, outputs, warnings)
    req = json.loads(Path(outputs["retraining_request"]).read_text())
    assert Path(outputs["retraining_request"]).parent == wd and req["new_data_path"] == str(wd / data.name)
    assert Path(outputs["new_data"]).is_file() and "concept_drift" in warnings[0]
    assert not (actions / "retraining_request.json").exists()
    assert len(list(actions.glob("retraining_request_*.handed.json"))) == 1
    assert not _gateway_request(target, wd, {}, [])          # déjà remise


def _report(i, drifted=0, degraded=False, labels=True):
    return {"window": {"index": i}, "dataset_drift": {"n_drifted": drifted},
            "performance": {"labels_available": labels, "realized_degraded": degraded}}


def test_new_data_is_the_current_episode_even_without_input_drift(tmp_path):
    # fenêtres 3-4 : ancien drift des entrées ; 5 : calme ; 6-7 : concept drift (perf mesurée en baisse)
    reports = [_report(i) for i in range(3)] + [_report(3, drifted=2), _report(4, drifted=1), _report(5),
                                                 _report(6, degraded=True), _report(7, degraded=True),
                                                 _report(8, labels=False)]
    asked = []
    store = SimpleNamespace(window_reports=lambda n: reports,
                            labelled_frame=lambda a, b, t: asked.append((a, b)) or pd.DataFrame(
                                {"x": [a], "target": [1], "y_pred": [0]}))
    fake = SimpleNamespace(history=50, store=store, target=SimpleNamespace(
        window_size=10, mapping=SimpleNamespace(features=["x"], target="target")))
    out = Supervisor.export_new_data(fake, str(tmp_path / "new.parquet"))
    assert asked == [(61, 70), (71, 80)]                      # seulement l'épisode 6-7
    assert list(pd.read_parquet(out).columns) == ["x", "target"]


def test_concept_scenario_is_a_learnable_rule():
    import random

    from monitoring_agent.runtime.simulator import perturb

    rng, med = random.Random(0), {"amount": 100.0}
    high = [perturb({"amount": 150.0}, "concept", rng, ["amount"], [], 1.0, med)[1] for _ in range(50)]
    low = [perturb({"amount": 50.0}, "concept", rng, ["amount"], [], 1.0, med)[1] for _ in range(50)]
    assert all(high) and not any(low)                         # au-delà de la médiane, la cible s'inverse
    assert perturb({"amount": 150.0}, "concept", rng, ["amount"], [], 0.0, med)[1] is False


def _target(tmp_path, monkeypatch, model_uri):
    from monitoring_agent.runtime.target import ColumnMapping, MonitoringTarget

    monkeypatch.setenv("MONITORING_RUNS_DIR", str(tmp_path / "runs"))
    mapping = ColumnMapping(task="classification", features=["x"], target="target", prediction="y_pred",
                            prediction_proba="y_pred_proba", timestamp="timestamp", id="request_id",
                            primary_metric="roc_auc", min_threshold=0.7)
    return MonitoringTarget(project="p", upstream_url="http://api", mapping=mapping, model_uri=model_uri)


def test_reference_is_the_training_data_of_the_deployed_model(tmp_path):
    from monitoring_agent.pipeline import _reference_data

    train = tmp_path / "train_v2.csv"
    train.write_text("x,target\n1,0\n")
    meta = tmp_path / "mlflow_metadata.json"
    meta.write_text(json.dumps({"model_uri": "models:/p/2", "params": {"training_data": str(train)}}))
    inp = SimpleNamespace(inputs={"mlflow_metadata": str(meta)})
    assert _reference_data(inp, SimpleNamespace(model_uri="models:/p/2")) == train
    assert _reference_data(inp, SimpleNamespace(model_uri="models:/p/1")) is None   # candidat non déployé
    assert _reference_data(SimpleNamespace(inputs={}), SimpleNamespace(model_uri="models:/p/2")) is None


def test_new_model_starts_a_new_monitoring_period(tmp_path, monkeypatch):
    from monitoring_agent.pipeline import _new_period

    old, new = _target(tmp_path, monkeypatch, "models:/p/1"), _target(tmp_path, monkeypatch, "unknown")
    new.dir.mkdir(parents=True)
    for suffix in ("", "-wal"):
        Path(f"{new.store_path}{suffix}").write_text("x")
    warnings = []
    _new_period(old, new, "models:/p/1", warnings)               # même modèle : rien ne bouge
    assert new.store_path.exists() and not warnings
    _new_period(old, new, "models:/p/2", warnings)
    assert not new.store_path.exists() and "archivé" in warnings[0]
    assert len(list(new.dir.glob("monitoring_p_v1_*.db"))) == 1 and len(list(new.dir.glob("monitoring_p_v1_*.db-wal"))) == 1


def test_concept_scenario_on_a_regression_target():
    """Régression : la cible des lignes concernées augmente (le marché change), elle ne devient pas « 1 - prix »."""
    import pandas as pd

    from monitoring_agent.runtime.simulator import CONCEPT_FACTOR, concept_label, is_regression_target

    prices = pd.Series([100.0 + i for i in range(50)])
    assert is_regression_target(prices) and not is_regression_target(pd.Series([0, 1] * 25))
    assert concept_label(200.0, regression=True) == 200.0 * CONCEPT_FACTOR
    assert concept_label(1, regression=False) == 0.0


class _Gateway:
    """Fausse gateway : statut, décisions, décision forcée et réponse humaine."""

    def __init__(self, target_dir, trigger=True, pending=None):
        self.dir, self.trigger, self.pending, self.calls = target_dir, trigger, pending or [], []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        body = self.pending if url.endswith("/decisions") else {}
        return type("R", (), {"json": lambda _self: body})()

    def post(self, url, timeout=None, json=None):
        self.calls.append(("POST", url))
        if url.endswith("/decide"):
            body = {"id": "D1", "status": "awaiting_human",
                    "policy": {"proposed_action": "request_retraining", "final_action": "request_retraining",
                               "mode": "human"}}
        else:                                    # approbation : la gateway écrit la demande
            actions = self.dir / "actions"
            actions.mkdir(parents=True, exist_ok=True)
            (actions / "retraining_request.json").write_text('{"reason": "performance_drop"}')
            body = {"id": "D1", "status": "executed"}
        return type("R", (), {"json": lambda _self: body})()


def _status(trigger):
    return {"project": "p", "agent": {"busy": False}, "last_window": {"window": {"index": 7}, "trigger_agent": trigger}}


def test_running_gateway_decides_instead_of_a_second_agent(tmp_path, monkeypatch):
    """La gateway tourne : son agent décide (il tient la mémoire Qdrant) ; la demande est remise à la ligne."""
    import httpx

    from monitoring_agent import pipeline

    target = type("T", (), {"dir": tmp_path / "p", "project": "p"})()
    fake = _Gateway(target.dir)
    monkeypatch.setattr(httpx, "get", fake.get)
    monkeypatch.setattr(httpx, "post", fake.post)
    outputs, warnings = {}, []
    assert pipeline._decide_in_gateway(target, _status(True), tmp_path, outputs, warnings)
    assert ("POST", "http://127.0.0.1:8100/monitoring/decide") in fake.calls
    assert ("POST", "http://127.0.0.1:8100/monitoring/decisions/D1/approve") in fake.calls
    assert "retraining_request" in outputs and "decision" in outputs


def test_gateway_calm_window_or_pending_decision(tmp_path, monkeypatch):
    import httpx

    from monitoring_agent import pipeline

    target = type("T", (), {"dir": tmp_path / "p", "project": "p"})()
    calm = _Gateway(target.dir)
    monkeypatch.setattr(httpx, "get", calm.get)
    monkeypatch.setattr(httpx, "post", calm.post)
    warnings = []
    assert not pipeline._decide_in_gateway(target, _status(False), tmp_path, {}, warnings)
    assert "sans signal" in warnings[0] and not any(c[0] == "POST" for c in calm.calls)

    waiting = _Gateway(target.dir, pending=[{"id": "D9", "status": "awaiting_human", "action": "request_retraining"}])
    monkeypatch.setattr(httpx, "get", waiting.get)
    warnings = []
    assert pipeline._decide_in_gateway(target, _status(True), tmp_path, {}, warnings)
    assert "D9" in warnings[0] and not any(c[0] == "POST" for c in waiting.calls)
