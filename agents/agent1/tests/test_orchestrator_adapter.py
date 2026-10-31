"""L'adaptateur respecte le contrat de l'orchestrateur (vrai validateur, sans appel au LLM)."""

import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.check_agent import check

ROOT = Path(__file__).resolve().parents[3]
PROJECTS = ROOT / "eval" / "projects"


@pytest.fixture(scope="module", autouse=True)
def _variants():
    """Les variantes (dont 09_no_manifest) sont générées, pas commitées."""
    if not (PROJECTS / "variants").exists():
        subprocess.run([sys.executable, str(ROOT / "eval" / "make_variants.py")], check=True)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("AGENT1_ENABLE_LLM", "false")
    monkeypatch.setenv("AGENT1_LLM_API_KEY", "")
    monkeypatch.setenv("ML_LLM_API_KEY", "")


def test_run_produces_the_contract_files(tmp_path):
    rep = check("agent1", "agents.agent1.orchestrator_adapter:run", PROJECTS / "credit_risk", tmp_path)
    assert rep.ok, [t for ok, t in rep.checks if not ok]


def test_project_without_manifest_gets_a_proposal(tmp_path):
    rep = check("propose_manifest", "agents.agent1.orchestrator_adapter:propose_manifest",
                PROJECTS / "variants" / "09_no_manifest", tmp_path)
    assert rep.ok, [t for ok, t in rep.checks if not ok]
    proposal = (tmp_path / "propose_manifest" / "proposed_manifest.json").read_text()
    assert '"target": "target"' in proposal and '"task": "classification"' in proposal
