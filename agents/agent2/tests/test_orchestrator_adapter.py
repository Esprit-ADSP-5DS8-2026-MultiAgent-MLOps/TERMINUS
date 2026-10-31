"""L'adaptateur respecte le contrat de l'orchestrateur (vrai validateur, MLflow temporaire, sans LLM)."""

from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("optuna")

from orchestrator.check_agent import check

ROOT = Path(__file__).resolve().parents[3]
CREDIT_RISK = ROOT / "eval" / "projects" / "credit_risk"
TARGET = "mlflow_agent.orchestrator_adapter:run"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.chdir(ROOT)  # le validateur est lancé avec « python -m validator » depuis la racine
    monkeypatch.setenv("AGENT2_NO_LLM", "1")
    monkeypatch.setenv("AGENT2_MAX_RUNS", "4")


def _meta(out: Path):
    from common.schemas import MlflowMetadata

    return MlflowMetadata.model_validate_json((out / "agent2" / "mlflow_metadata.json").read_text())


def test_candidate_is_registered_and_asks_for_promotion(tmp_path):
    rep = check("agent2", TARGET, CREDIT_RISK, tmp_path / "run1")
    assert rep.ok, [t for ok, t in rep.checks if not ok]
    meta = _meta(tmp_path / "run1")
    assert meta.model_uri == "models:/credit_risk/1" and meta.alias == "candidate"
    assert meta.quality_gate.passed and meta.champion_replacement_requested
    assert meta.quality_gate.value > meta.quality_gate.baseline_value


def test_same_model_does_not_replace_the_champion(tmp_path):
    from mlflow_agent.orchestrator_adapter import promote

    assert check("agent2", TARGET, CREDIT_RISK, tmp_path / "run1").ok
    promote("credit_risk", _meta(tmp_path / "run1").model_version)
    rep = check("agent2", TARGET, CREDIT_RISK, tmp_path / "run2")   # mêmes données, même graine
    assert rep.ok
    gate = _meta(tmp_path / "run2").quality_gate
    assert gate.champion_value is not None and not gate.passed
    assert any("champion" in r for r in gate.reasons)
