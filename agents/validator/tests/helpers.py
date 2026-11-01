"""Outils de test partagés : faux LLM scripté, faux sandbox, fabrique de rapports."""
import json
from pathlib import Path

from agentic_core.schemas import AdmissionStatus, Issue, Nature, ValidatorReport, seal
from policy import load_contract
from validator.handoff import compute_project_hash

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "eval" / "projects"
CREDIT = PROJECTS / "credit_risk"
VARIANTS = PROJECTS / "variants"


def tc(name, args, i=1):
    return {"content": "", "tool_calls": [{"id": f"c{i}", "function": {"name": name, "arguments": json.dumps(args)}}]}


def calls(*pairs):
    """calls(("list_files", {}), ("submit_report", {...})) -> script de messages assistant."""
    return [tc(n, a, i) for i, (n, a) in enumerate(pairs, 1)]


class FakeRunner:
    """Faux SandboxRunner : aucune exécution réelle, résultats canned."""
    info = {"backend": "fake", "network_isolated": True,
            "environment": {"mode": "venv", "backend": "fake", "python": "/fake/venv/bin/python", "is_virtualenv": True,
                            "shares_validator_env": False, "own_venv": True, "deps_source": "requirements.txt", "pinned_checked": 4,
                            "mismatches": [], "missing": [], "network_isolated": True, "verified": True}}

    def __init__(self, harness=None, others=None, env="venv"):
        self.harness = harness or {"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie"}
        self.others = others or {}
        self.runs = []
        self.env = env

    def run(self, command, root=None, **p):
        self.runs.append((command, str(root), p))
        if command == "harness":
            return dict(self.harness, command="harness")
        return self.others.get(command, {"command": command, "ok": True, "returncode": 0})


def all_pass(contract, **override):
    return [{"check_id": i, "status": "PASS", **override.get(i, {})} for i in contract.mandatory_ids]


def validator_report(project: Path, status=AdmissionStatus.REPAIR_REQUIRED, issues=None, **kw):
    h, n = compute_project_hash(project)
    issues = issues if issues is not None else [Issue(issue_id="I1", check_id="ENV.DEPENDENCIES_PINNED", category="dependencies",
                                                      description="dépendance non épinglée", nature=Nature.CONFIGURATION)]
    return seal(ValidatorReport(report_id="v-test", project=project.name, project_hash=h, hashed_files=n, status=status, issues=issues,
                                contract_id="sklearn-tabular", **kw))
