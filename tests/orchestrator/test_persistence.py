"""
Tests de la persistance : l'état est sauvegardé dans runs/checkpoints.sqlite.
"""

from pathlib import Path

import pytest

from orchestrator import graph, nodes
from orchestrator.__main__ import main
from orchestrator.graph import Run, RunNotFound, list_runs

ROOT = Path(__file__).resolve().parents[2]
CREDIT_RISK = str(ROOT / "eval" / "projects" / "credit_risk")
APPROVE = {"decision": "approve", "by": "pytest", "comment": ""}


@pytest.fixture(autouse=True)
def no_drift(monkeypatch):
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)


def test_paused_run_survives_and_can_be_approved_later(tmp_path):
    run = Run.start(CREDIT_RISK, runs_dir=str(tmp_path))
    assert run.status == "WAITING_HUMAN"
    run_id = run.run_id
    del run  # on « ferme le terminal »

    reloaded = Run.load(run_id, runs_dir=str(tmp_path))  # nouvel objet, relu depuis SQLite
    assert reloaded.question["gate"] == "champion_replacement"
    reloaded.answer(APPROVE)
    assert reloaded.status == "DONE"
    assert (tmp_path / run_id / "run_summary.json").exists()


def test_crashed_run_resumes_without_rerunning_the_validator(tmp_path, monkeypatch):
    calls = []
    real_validator = nodes.run_validator
    monkeypatch.setattr(nodes, "run_validator", lambda *a: calls.append(1) or real_validator(*a))

    def crash(inp):
        raise KeyboardInterrupt("coupure simulée pendant le déploiement")

    monkeypatch.setitem(graph.AGENTS, "agent3", crash)
    run = Run.start(CREDIT_RISK, runs_dir=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        run.answer(APPROVE)

    monkeypatch.setitem(graph.AGENTS, "agent3", graph.agent3_stub)  # la panne est réparée
    reloaded = Run.load(run.run_id, runs_dir=str(tmp_path))
    assert reloaded.status == "INTERRUPTED"
    assert reloaded.snapshot.next == ("agent3",)

    reloaded.resume()
    assert reloaded.status == "DONE"
    assert len(calls) == 1  # le validateur n'a pas été relancé


def test_unknown_run_is_reported(tmp_path):
    Run.start(CREDIT_RISK, runs_dir=str(tmp_path))  # crée la base
    with pytest.raises(RunNotFound):
        Run.load("inexistant", runs_dir=str(tmp_path))


def test_cli_no_wait_then_approve(tmp_path, capsys):
    runs = str(tmp_path)
    assert main(["--runs-dir", runs, "run", CREDIT_RISK, "--no-wait"]) == 0
    [run] = list_runs(runs)
    assert run.status == "WAITING_HUMAN"

    assert main(["--runs-dir", runs, "status"]) == 0
    assert run.run_id in capsys.readouterr().out

    assert main(["--runs-dir", runs, "approve", run.run_id, "-m", "ok"]) == 0
    assert Run.load(run.run_id, runs).status == "DONE"
    assert main(["--runs-dir", runs, "approve", run.run_id]) == 2  # plus rien à approuver


# --- choix de la base -------------------------------------------------------

from orchestrator import checkpoint  # noqa: E402


def test_sqlite_is_used_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_DATABASE_URL", raising=False)
    assert not checkpoint.uses_postgres()
    assert checkpoint.describe_backend(tmp_path).startswith("SQLite")


def test_postgres_url_is_described_without_password(monkeypatch):
    monkeypatch.setenv("ORCH_DATABASE_URL", "postgresql://mlops:secret@db.local:5432/orchestrator")
    assert checkpoint.uses_postgres()
    desc = checkpoint.describe_backend()
    assert desc == "PostgreSQL · db.local:5432/orchestrator"
    assert "secret" not in desc


def test_unreachable_postgres_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_DATABASE_URL", "postgresql://mlops@localhost:1/orchestrator")
    assert main(["--runs-dir", str(tmp_path), "run", CREDIT_RISK, "--no-wait"]) == 2


def test_resumed_run_keeps_the_agents_it_started_with(tmp_path, monkeypatch):
    """Lancé avec les stubs, repris dans un terminal où un vrai agent est actif : il garde les stubs."""
    run = Run.start(CREDIT_RISK, runs_dir=str(tmp_path))
    assert run.status == "WAITING_HUMAN"
    calls = []

    def real_agent4(inp):
        calls.append(inp.run_id)
        raise AssertionError("le vrai agent ne doit pas tourner dans ce run")

    monkeypatch.setitem(graph.AGENT_ORIGIN, "agent4", "réel")    # « nouveau terminal » : agent4 réel
    monkeypatch.setitem(graph.AGENTS, "agent4", real_agent4)
    reloaded = Run.load(run.run_id, runs_dir=str(tmp_path))
    reloaded.answer(APPROVE)
    assert reloaded.status == "DONE" and not calls
