"""
Boucle validateur -> réparation -> juge -> revalidation (nœud réservé au validateur v2).

L'agent de réparation est simulé : il corrige une vraie variante du projet (dépendance non épinglée),
et le vrai validateur revalide la copie.
"""

import shutil
from pathlib import Path

import pytest

from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    JudgeVerdict,
    RepairChange,
    RepairResult,
    write_output,
)
from orchestrator import graph
from orchestrator.graph import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
UNPINNED = ROOT / "eval" / "projects" / "variants" / "02_unpinned_deps"


def approve(question):
    return {"decision": "approve", "by": "pytest", "comment": ""}


def fake_repair(fix: bool = True, approved: bool = True, outside: Path | None = None):
    """Faux agent de réparation : copie le projet dans son workdir et épingle scikit-learn."""

    def run(inp: AgentInput) -> AgentResult:
        wd = Path(inp.workdir)
        copy = outside or wd / "project"
        shutil.copytree(inp.project_dir, copy, dirs_exist_ok=True)
        if fix:
            req = copy / "requirements.txt"
            req.write_text(req.read_text().replace("scikit-learn>=1.0", "scikit-learn==1.8.0"))
        rr = RepairResult(
            attempt=int(wd.name.split("_")[-1]),
            source_report=inp.report_path,
            repaired_project=str(copy),
            changes=[RepairChange(file="requirements.txt", finding_code="DEPS.UNPINNED",
                                  description="scikit-learn épinglé en 1.8.0")],
            judge=JudgeVerdict(approved=approved, score=0.9 if approved else 0.2,
                               rationale="version alignée sur l'environnement de référence"),
        )
        return AgentResult(agent="repair", status=AgentStatus.OK, can_proceed=True,
                           outputs={"repair_result": write_output(inp, "repair_result", rr),
                                    "repaired_project": str(copy)})

    return run


@pytest.fixture(autouse=True)
def _need_variant():
    if not UNPINNED.exists():
        pytest.skip("variante 02_unpinned_deps absente : lancer python eval/make_variants.py")


def steps(s):
    return [h["node"] for h in s["history"]]


def test_repaired_project_is_revalidated_and_deployed(tmp_path, monkeypatch):
    monkeypatch.setitem(graph.AGENTS, "repair", fake_repair())
    s = run_pipeline(str(UNPINNED), approve, runs_dir=str(tmp_path), simulate={})
    assert s["final_status"] == "DONE"
    assert steps(s)[:3] == ["validator", "repair", "validator"]
    assert s["repair_count"] == 1
    assert Path(s["repaired_dir"]).is_relative_to(tmp_path)
    # le projet d'origine n'a pas été touché
    assert ">=" in (UNPINNED / "requirements.txt").read_text()
    # les agents suivants travaillent sur la copie réparée
    assert (tmp_path / s["run_id"] / "validator_after_repair_1" / "compatibility_report.json").exists()


def test_repair_stops_after_two_attempts(tmp_path, monkeypatch):
    monkeypatch.setitem(graph.AGENTS, "repair", fake_repair(fix=False))  # « répare » sans rien corriger
    s = run_pipeline(str(UNPINNED), approve, runs_dir=str(tmp_path), simulate={})
    assert s["final_status"] == "BLOCKED"
    assert steps(s) == ["validator", "repair", "validator", "repair", "validator"]


def test_repair_rejected_by_the_judge_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setitem(graph.AGENTS, "repair", fake_repair(approved=False))
    s = run_pipeline(str(UNPINNED), approve, runs_dir=str(tmp_path), simulate={})
    assert s["final_status"] == "BLOCKED"
    assert steps(s) == ["validator", "repair"]
    assert "juge" in s["results"]["repair"]["reasons"][-1]
    assert "repaired_dir" not in s


def test_repair_written_outside_its_workdir_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(graph.AGENTS, "repair", fake_repair(outside=tmp_path / "ailleurs"))
    s = run_pipeline(str(UNPINNED), approve, runs_dir=str(tmp_path / "runs"), simulate={})
    assert s["final_status"] == "BLOCKED"
    assert "workdir" in s["results"]["repair"]["reasons"][-1]
