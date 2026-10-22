"""
Tests de bout en bout de l'orchestrateur, avec le VRAI validateur et les agents stubs.
"""

from pathlib import Path

import pytest

from orchestrator.graph import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / "eval" / "projects"


def approve(question):
    return {"decision": "approve", "by": "pytest", "comment": ""}


def reject(question):
    return {"decision": "reject", "by": "pytest", "comment": ""}


def test_accepted_project_goes_to_the_end(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)
    s = run_pipeline(str(PROJECTS / "credit_risk"), approve, runs_dir=str(tmp_path))
    assert s["final_status"] == "DONE"
    assert list(s["results"]) == ["validator", "agent1", "agent2", "agent3", "agent4"]
    for key in ("project_spec", "data_profile", "baseline", "mlflow_metadata", "deployment_result"):
        assert Path(s["artifacts"][key]).exists(), key


def test_human_can_refuse_the_champion(tmp_path):
    s = run_pipeline(str(PROJECTS / "credit_risk"), reject, runs_dir=str(tmp_path))
    assert s["final_status"] == "REJECTED"
    assert "agent3" not in s["results"]  # rien n'est déployé


def test_drift_triggers_one_retraining_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_SIMULATE_DRIFT", "1")
    s = run_pipeline(str(PROJECTS / "credit_risk"), approve, runs_dir=str(tmp_path))
    gates = [d["gate"] for d in s["decisions"]]
    assert gates == ["champion_replacement", "retraining", "champion_replacement"]
    assert s["retrain_count"] == 1
    assert s["final_status"] == "DONE"
    steps = [h["node"] for h in s["history"]]  # chaque exécution est conservée
    assert steps == ["validator", "agent1", "agent2", "agent3", "agent4", "agent2", "agent3", "agent4"]


def test_rejected_project_stops_after_validator(tmp_path):
    variants = sorted((PROJECTS / "variants").glob("02_*"))
    if not variants:
        pytest.skip("variante 02_* introuvable : lancer python eval/make_variants.py")
    s = run_pipeline(str(variants[0]), approve, runs_dir=str(tmp_path))
    assert s["final_status"] == "BLOCKED"
    assert list(s["results"]) == ["validator"]
