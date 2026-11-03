"""API de Terminus avec les stubs et le vrai validateur (commandes synchrones, sans thread)."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from orchestrator.api import create_app
from orchestrator.runlock import RunBusy, is_busy, run_lock


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)
    return TestClient(create_app(str(tmp_path / "runs"), sync=True))


def _start(api) -> str:
    r = api.post("/api/runs", json={"project_dir": "eval/projects/credit_risk"})
    assert r.status_code == 202
    return r.json()["run_id"]


def test_line_waits_at_the_champion_signal_then_is_approved(api):
    run_id = _start(api)
    d = api.get(f"/api/runs/{run_id}").json()
    assert d["status"] == "WAITING_HUMAN" and d["gate"]["gate"] == "champion_replacement"
    track = [(s["id"], s["status"]) for s in d["track"]]
    assert track[:3] == [("validator", "passed"), ("agent1", "passed"), ("agent2", "passed")]
    assert ("gate_champion", "waiting") in track and track[-1] == ("agent4", "todo")
    assert d["gate"]["content"]["quality_gate"]["passed"]

    stale = {"decision": "approve", "by": "Mohamed", "key": "champion_replacement:9"}
    assert api.post(f"/api/runs/{run_id}/decision", json=stale).status_code == 409
    ok = {"decision": "approve", "by": "Mohamed", "key": d["gate"]["key"], "publish": False}
    assert api.post(f"/api/runs/{run_id}/decision", json=ok).status_code == 202
    d = api.get(f"/api/runs/{run_id}").json()
    assert d["status"] == "DONE" and d["decisions"][-1]["by"] == "Mohamed (Terminus)"
    assert ("gate_champion", "passed") in [(s["id"], s["status"]) for s in d["track"]]
    assert not any(s["status"] == "todo" for s in d["track"])
    assert json.loads((Path(d["project_dir"]) / "mlops.yaml").exists().__str__().lower())


def test_lists_runs_decisions_files_and_projects(api, tmp_path):
    run_id = _start(api)
    assert api.get("/api/runs").json()[0]["run_id"] == run_id
    pending = api.get("/api/decisions").json()["pending"]
    assert pending[0]["gate"] == "champion_replacement" and pending[0]["run_id"] == run_id
    detail = api.get(f"/api/runs/{run_id}").json()
    assert detail["artifacts"]["mlflow_metadata"].startswith("agent2/")
    assert "quality_gate" in api.get(f"/api/runs/{run_id}/file", params={"path": "agent2/mlflow_metadata.json"}).text
    assert api.get(f"/api/runs/{run_id}/file", params={"path": "../../../etc/passwd"}).status_code == 404
    assert any(p["name"] == "credit_risk" for p in api.get("/api/projects").json())
    assert api.post("/api/runs", json={"project_dir": "nulle/part"}).status_code == 400


def test_decision_refused_while_the_line_runs_elsewhere(api, tmp_path):
    run_id = _start(api)
    key = api.get(f"/api/runs/{run_id}").json()["gate"]["key"]
    with run_lock(tmp_path / "runs" / run_id):
        assert is_busy(tmp_path / "runs" / run_id)
        r = api.post(f"/api/runs/{run_id}/decision", json={"decision": "approve", "by": "x", "key": key})
        assert r.status_code == 409
        with pytest.raises(RunBusy), run_lock(tmp_path / "runs" / run_id):
            pass
    assert not is_busy(tmp_path / "runs" / run_id)


def test_health_never_returns_secrets(api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-123")
    body = api.get("/api/health").text
    assert "secret-123" not in body and "présente" in body


def test_unknown_line_and_monitor_on_undeployed_line(api):
    assert api.get("/api/runs/inconnu").status_code == 404
    run_id = _start(api)
    assert api.post(f"/api/runs/{run_id}/monitor").status_code == 409


def test_files_listed_when_runs_dir_is_relative(tmp_path, monkeypatch):
    """`python -m orchestrator api` passe « runs » (relatif) : les fichiers doivent rester listés."""
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)
    api = TestClient(create_app(os.path.relpath(tmp_path / "runs"), sync=True))
    run_id = _start(api)
    artifacts = api.get(f"/api/runs/{run_id}").json()["artifacts"]
    assert artifacts["mlflow_metadata"] == "agent2/mlflow_metadata.json"


def test_line_being_formed_has_the_full_shape(api, tmp_path):
    """Juste après « lancer la ligne », l'interface reçoit déjà un plan complet (sinon page blanche)."""
    (tmp_path / "runs" / "neuve").mkdir(parents=True)
    d = api.get("/api/runs/neuve").json()
    assert d["status"] == "STARTING" and d["supervision"] == [] and d["artifacts"] == {} and d["gate"] is None
    assert [s["status"] for s in d["track"]][:2] == ["running", "todo"]


def test_registry_history_shows_the_champion_alias(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    from orchestrator import api as api_module
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    c = MlflowClient()
    c.create_registered_model("projet")
    for _ in range(2):
        c.create_model_version("projet", source=str(tmp_path))
    c.set_registered_model_alias("projet", "champion", "2")
    history = api_module.model_history("projet")
    assert [(v["version"], v["aliases"]) for v in history] == [(2, ["champion"]), (1, [])]
