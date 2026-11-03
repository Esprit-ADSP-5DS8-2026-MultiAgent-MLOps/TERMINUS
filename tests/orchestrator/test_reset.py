"""Remise à zéro d'un projet : seuls ses runs partent, archivés ; les autres projets restent."""

import os
from pathlib import Path

import pytest

from orchestrator import reset
from orchestrator.checkpoint import get_checkpointer
from orchestrator.graph import Run, RunNotFound


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)
    monkeypatch.delenv("ORCH_DATABASE_URL", raising=False)
    monkeypatch.setenv("MONITORING_RUNS_DIR", str(tmp_path / "agent4"))
    monkeypatch.setattr(reset, "_has_model", lambda project: False)
    monkeypatch.setattr(reset, "_has_container", lambda project: False)
    return tmp_path


def test_reset_archives_only_this_project(env):
    runs = str(env / "runs")
    keep = Run.start("eval/projects/credit_risk", runs, supervisor="rules", run_id="credit1")
    (env / "agent4" / "credit_risk").mkdir(parents=True)
    (env / "runs" / "mlflow").mkdir()                        # pas un run : ignoré
    other = env / "runs" / "autre"
    other.mkdir()

    p = reset.plan("credit_risk", runs)
    assert p.runs == ["credit1"] and p.monitoring == env / "agent4" / "credit_risk"
    assert reset.plan("autre_projet", runs).empty()

    archive = reset.apply(p, runs, archive=env / "archive")
    assert (archive / "runs" / "credit1").is_dir() and (archive / "agent4").is_dir()
    assert not (env / "runs" / "credit1").exists() and other.exists()
    with pytest.raises(RunNotFound):
        Run.load(keep.run_id, runs, get_checkpointer(runs))
    assert reset.plan("credit_risk", runs).empty()


def test_dry_run_changes_nothing(env, capsys):
    runs = str(env / "runs")
    Run.start("eval/projects/credit_risk", runs, supervisor="rules", run_id="credit2")
    assert reset.main("credit_risk", runs, yes=False) == 0
    assert "--yes" in capsys.readouterr().out
    assert Path(runs, "credit2").is_dir()


def test_relative_runs_dir_starts_from_the_repo(env, monkeypatch):
    """Lancée depuis agents/agent4, la commande doit voir les runs de la racine, pas ceux de l'Agent 4."""
    monkeypatch.setattr(reset, "REPO", env)
    (env / ".env").write_text("RESET_TEST_VAR=racine\n")
    monkeypatch.delenv("RESET_TEST_VAR", raising=False)
    elsewhere = env / "agents" / "agent4"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    assert reset.repo_context("runs") == str(env / "runs")
    assert os.environ["RESET_TEST_VAR"] == "racine"
