"""Chaîne complète AVEC exécution réelle dans le sandbox (LLM scripté) : Validator -> Repair -> Judge -> revalidation.

Désactivé par défaut (lent ; construit des venvs si le réseau est disponible). Activer :
    CV_RUN_SLOW=1 pytest tests/test_integration_slow.py                       # venv construit depuis requirements.txt (réseau)
    CV_RUN_SLOW=1 CV_PYTHON=/chemin/python-avec-sklearn pytest tests/test_integration_slow.py   # sans réseau
"""
import json
import os

import pytest
from helpers import CREDIT, VARIANTS, calls

from agentic_core.handoff import verify_report
from agentic_core.llm_client import ScriptedClient
from agentic_core.schemas import AdmissionStatus as S
from repair_agent.judge import run_judge
from policy import load_contract
from repair_agent import RepairConfig, run_repair
from validator_agent import ValidatorConfig, run_validator

pytestmark = pytest.mark.slow
ENV = os.environ.get("CV_PYTHON", "venv")
if not os.environ.get("CV_RUN_SLOW"):
    pytest.skip("définir CV_RUN_SLOW=1 pour exécuter", allow_module_level=True)

C0 = load_contract()
HARNESS = ("run_in_sandbox", {"command": "harness", "data_path": "data.csv", "target": "target", "primary_metric": "roc_auc"})
SPEC = {"task": "classification", "target": "target", "primary_metric": "roc_auc", "min_threshold": 0.7, "data_path": "data.csv"}


def checks(**o):
    return [{"check_id": i, "status": "PASS", **o.get(i, {})} for i in C0.mandatory_ids]


def sub(chk, issues=()):
    return ("submit_report", {"project_type": "tabular_ml", "checks": chk, "issues": list(issues), "project_spec": SPEC})


def test_real_execution_accepts_the_reference_project():
    r = run_validator(CREDIT, ScriptedClient(calls(("list_files", {}), ("analyze_code", {}), HARNESS, sub(checks()))), C0, ValidatorConfig(env=ENV))
    assert r.status == S.ACCEPTED and r.observed_metrics["roc_auc"] > 0.6 and r.sandbox


def test_full_chain_repairs_hardcoded_path_and_revalidates(tmp_path):
    src = VARIANTS / "04_hardcoded_path"
    ev = 'BACKUP = "/home/user/credit.csv"'
    chk = checks(**{"IO.PATHS": {"status": "FAIL", "evidence": ev, "file": "train.py", "line": 14}})
    issue = {"check_id": "IO.PATHS", "category": "code", "nature": "mechanical", "file": "train.py", "line": 14, "description": "chemin absolu en dur", "evidence": ev}
    v1 = run_validator(src, ScriptedClient(calls(("read_file", {"path": "train.py"}), ("analyze_code", {}), HARNESS, sub(chk, [issue]))), C0, ValidatorConfig(env=ENV))
    assert v1.status == S.REPAIR_REQUIRED and v1.issues[0].file == "train.py"

    fix = calls(("replace_in_file", {"path": "train.py", "old_str": ev, "new_str": 'BACKUP = "data.csv"', "issue_id": "I1", "reason": "chemin relatif"}),
                ("run_checks", {}), ("submit_repair", {"results": [{"issue_id": "I1", "outcome": "fixed", "patch_ids": ["p0001"], "explanation": "chemin relatif"}]}))
    rr = run_repair(src, v1, tmp_path / "fixed", ScriptedClient(fix), RepairConfig(env=ENV))
    assert rr.status == "REPAIRED" and not rr.requires_human_approval and rr.next_action == "JUDGE"
    assert {t.name: t.status for t in rr.tests}["import"] == "PASS" and {t.name: t.status for t in rr.tests}["static"] == "PASS"
    assert rr.metrics_comparison.status in ("COMPARED", "AFTER_ONLY")

    verdict_ev = "BACKUP"
    crit = [{"name": n, "result": "PASS", "rationale": "ok", "evidence": verdict_ev} for n in ("correctness", "completeness", "regression_risk", "evidence_quality", "scope_compliance")]
    jv = run_judge(src, rr, ScriptedClient(calls(("list_changes", {}), ("get_diff", {}), ("get_test_results", {}), ("submit_verdict",
                   {"verdict": "ACCEPT", "next_action": "REVALIDATE", "reason": "ok", "criteria": crit})), model="judge-x"), v1)
    assert jv.verdict == "ACCEPT"

    v2 = run_validator(rr.workspace_path, ScriptedClient(calls(("list_files", {}), ("analyze_code", {}), HARNESS, sub(checks()))), C0, ValidatorConfig(env=ENV), parent=jv)
    assert v2.status == S.ACCEPTED
    assert [x.report_type for x in v2.lineage] == ["validator", "repair", "judge"]
    assert verify_report(v2, rr.workspace_path).ok
