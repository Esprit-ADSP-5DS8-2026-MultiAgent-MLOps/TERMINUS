"""Le nœud « repair » de l'orchestrateur : Validator v2 -> Repair -> Judge à travers l'adaptateur (LLM scriptés).

Le projet est réellement exécuté dans le sandbox, dans un venv construit depuis son requirements.txt.
"""
import json
from pathlib import Path

import pytest
from agentic_core.llm_client import ScriptedClient
from helpers import calls
from policy import load_contract

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "eval" / "projects" / "variants" / "04_hardcoded_path"
EV = 'BACKUP = "/home/user/credit.csv"'
C0 = load_contract()
HARNESS = ("run_in_sandbox", {"command": "harness", "data_path": "data.csv", "target": "target", "primary_metric": "roc_auc"})
SPEC = {"task": "classification", "target": "target", "primary_metric": "roc_auc", "min_threshold": 0.7, "data_path": "data.csv"}
CRIT = ("correctness", "completeness", "regression_risk", "evidence_quality", "scope_compliance")


def _validator():
    chk = [{"check_id": i, "status": "PASS"} for i in C0.mandatory_ids]
    for c in chk:
        if c["check_id"] == "IO.PATHS":
            c.update(status="FAIL", evidence=EV, file="train.py", line=14)
    issue = {"check_id": "IO.PATHS", "category": "code", "nature": "mechanical", "file": "train.py", "line": 14,
             "description": "chemin absolu en dur", "evidence": EV}
    return calls(("read_file", {"path": "train.py"}), ("analyze_code", {}), HARNESS,
                 ("submit_report", {"project_type": "tabular_ml", "checks": chk, "issues": [issue], "project_spec": SPEC}))


def _repair():
    return calls(("replace_in_file", {"path": "train.py", "old_str": EV, "new_str": 'BACKUP = "data.csv"', "issue_id": "I1",
                                      "reason": "chemin relatif"}),
                 ("run_checks", {}),
                 ("submit_repair", {"results": [{"issue_id": "I1", "outcome": "fixed", "patch_ids": ["p0001"],
                                                 "explanation": "chemin relatif"}]}))


def _judge(verdict):
    crit = [{"name": n, "result": "PASS" if verdict == "ACCEPT" else "CONCERN", "rationale": "ok", "evidence": "BACKUP"}
            for n in CRIT]
    action = "REVALIDATE" if verdict == "ACCEPT" else "RETRY"
    return calls(("list_changes", {}), ("get_diff", {}), ("get_test_results", {}),
                 ("submit_verdict", {"verdict": verdict, "next_action": action, "reason": "chemin corrigé", "criteria": crit}))


@pytest.fixture
def scripted(monkeypatch):
    def use(verdict="ACCEPT"):
        scripts = {"validator": _validator(), "repair": _repair(), "judge": _judge(verdict)}
        monkeypatch.setattr("agentic_core.llm_client.build_client",
                            lambda agent, *a, **k: ScriptedClient(scripts[agent], model=f"{agent}-scripte"))
    monkeypatch.setenv("REPAIR_ENV", "venv")          # venv isolé (pip install : réseau requis la 1re fois)
    monkeypatch.setenv("REPAIR_SANDBOX", "subprocess")
    monkeypatch.chdir(REPO)
    return use


def test_accepted_repair_gives_a_copy_to_revalidate(tmp_path, scripted):
    from orchestrator.check_agent import check

    scripted("ACCEPT")
    rep = check("repair", "repair_agent.orchestrator_adapter:run", PROJECT, tmp_path / "out")
    assert rep.ok, [t for ok, t in rep.checks if not ok]
    rr = json.loads((tmp_path / "out" / "repair" / "repair_result.json").read_text())
    assert rr["judge"]["approved"] and rr["changes"][0]["finding_code"] == "IO.PATHS"
    fixed = Path(rr["repaired_project"])
    assert fixed.name == PROJECT.name                         # project_name inchangé pour MLflow et Docker
    assert 'BACKUP = "data.csv"' in (fixed / "train.py").read_text()
    assert EV in (PROJECT / "train.py").read_text()          # le projet d'origine n'a pas bougé


def test_revise_verdict_is_not_used(tmp_path, scripted):
    from repair_agent.orchestrator_adapter import run

    from orchestrator.check_agent import prepare

    scripted("REVISE")
    result = run(prepare("repair", PROJECT, tmp_path / "out"))
    assert not result.can_proceed and "REVISE" in result.reasons[0]


def test_agent_env_is_restored(tmp_path, scripted, monkeypatch):
    """Les clés de agents/validator/.env ne restent pas dans le processus de l'orchestrateur après la chaîne."""
    import os

    from common.envscope import agent_env

    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=cle-de-sarra\n")
    monkeypatch.setenv("GROQ_API_KEY", "cle-de-mohamed")
    with agent_env(env_file):
        assert os.environ["GROQ_API_KEY"] == "cle-de-sarra"
        os.environ["AJOUT_DE_L_AGENT"] = "x"
    assert os.environ["GROQ_API_KEY"] == "cle-de-mohamed" and "AJOUT_DE_L_AGENT" not in os.environ


def test_pipeline_repairs_revalidates_and_goes_on(tmp_path, scripted, monkeypatch):
    """Orchestrateur complet : v1 bloque -> réparation acceptée -> le v1 revalide la copie -> la suite du pipeline."""
    from repair_agent.orchestrator_adapter import run

    from orchestrator import graph

    scripted("ACCEPT")
    monkeypatch.setitem(graph.AGENTS, "repair", run)
    s = graph.run_pipeline(str(PROJECT), lambda q: {"decision": "approve", "by": "pytest", "comment": ""},
                           runs_dir=str(tmp_path / "runs"), simulate={})
    steps = [h["node"] for h in s["history"]]
    assert steps[:3] == ["validator", "repair", "validator"], steps
    assert s["final_status"] == "DONE" and s["results"]["validator"]["status"] in ("OK", "WARNING")
    assert EV in (PROJECT / "train.py").read_text()
