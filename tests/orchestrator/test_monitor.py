"""`monitor` : après un déploiement, l'Agent 4 relit la production ; une demande de réentraînement
repasse par les gates humaines, l'Agent 2, puis l'Agent 3 (stubs, vrai validateur)."""

import json
from pathlib import Path

import pytest

from orchestrator.graph import Run, run_pipeline
from orchestrator.stubs.agents import SIMULATE_FILE

ROOT = Path(__file__).resolve().parents[2]
CREDIT_RISK = str(ROOT / "eval" / "projects" / "credit_risk")


def approve(question):
    return {"decision": "approve", "by": "pytest", "comment": ""}


def _deployed(tmp_path, monkeypatch) -> Run:
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path))
    assert s["final_status"] == "DONE"
    return Run.load(s["run_id"], str(tmp_path))


def _drift(run: Run) -> None:
    (run.run_dir / SIMULATE_FILE).write_text(json.dumps({"drift": True, "failure": None}))


def test_nothing_to_retrain_ends_done_again(tmp_path, monkeypatch):
    run = _deployed(tmp_path, monkeypatch)
    run.monitor()
    assert run.status == "DONE"
    assert [h["node"] for h in run.state["history"]][-1] == "agent4"
    assert sum(h["node"] == "agent4" for h in run.state["history"]) == 2


def test_retraining_request_goes_through_the_gates(tmp_path, monkeypatch):
    run = _deployed(tmp_path, monkeypatch)
    _drift(run)
    run.monitor()
    assert run.status == "WAITING_HUMAN" and run.question["gate"] == "retraining"
    run.answer(approve(run.question))                       # réentraînement approuvé -> Agent 2
    assert run.question["gate"] == "champion_replacement"
    run.answer(approve(run.question))                       # nouveau champion -> Agent 3 -> Agent 4
    assert run.status == "DONE" and run.state["retrain_count"] == 1
    nodes = [h["node"] for h in run.state["history"]]
    assert nodes.count("agent2") == 2 and nodes.count("agent3") == 2 and nodes.count("agent4") == 3
    gates = [d["gate"] for d in run.state["decisions"]]
    assert gates == ["champion_replacement", "retraining", "champion_replacement"]

    run.monitor()                                           # nouvel épisode : la gate est reposée
    assert run.question["gate"] == "retraining"
    run.answer({"decision": "reject", "by": "pytest", "comment": "pas maintenant"})
    assert run.status == "DONE"


def test_only_a_deployed_run_can_be_monitored(tmp_path):
    s = run_pipeline(CREDIT_RISK, lambda q: {"decision": "reject", "by": "pytest", "comment": ""},
                     runs_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="DONE"):
        Run.load(s["run_id"], str(tmp_path)).monitor()


def test_rejected_retraining_candidate_keeps_the_deployed_model():
    from orchestrator.routing import after_agent2

    failed_gate = {"status": "FAILED", "can_proceed": False, "outputs": {"mlflow_metadata": "m.json"}}
    crashed = {"status": "FAILED", "can_proceed": False, "outputs": {}}
    assert after_agent2({"results": {"agent2": failed_gate}, "retrain_count": 1}) == "end_champion_kept"
    assert after_agent2({"results": {"agent2": failed_gate}, "retrain_count": 0}) == "end_failed"
    assert after_agent2({"results": {"agent2": crashed}, "retrain_count": 1}) == "end_failed"
