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
    training = Path(meta.params["training_data"])             # référence du monitoring de l'Agent 4
    assert training.name == "train_v1.csv" and training.is_file()


def test_same_model_does_not_replace_the_champion(tmp_path):
    from mlflow_agent.orchestrator_adapter import promote

    assert check("agent2", TARGET, CREDIT_RISK, tmp_path / "run1").ok
    promote("credit_risk", _meta(tmp_path / "run1").model_version)
    rep = check("agent2", TARGET, CREDIT_RISK, tmp_path / "run2")   # mêmes données, même graine
    assert rep.ok
    gate = _meta(tmp_path / "run2").quality_gate
    assert gate.champion_value is not None and not gate.passed
    assert gate.champion_value == pytest.approx(gate.value)           # rejugé sur le même jeu de test
    assert _meta(tmp_path / "run2").params["label_classes"] == "[0, 1]"
    assert any("champion" in r for r in gate.reasons)
    assert "AgentResult valide (statut WARNING, can_proceed=False)" in rep.checks[0][1]
    from mlflow_agent.orchestrator_adapter import _client
    candidate = _client().get_model_version_by_alias("credit_risk", "candidate").version
    assert str(candidate) == _meta(tmp_path / "run1").model_version  # l'alias reste sur le candidat accepté


def test_candidate_worse_than_the_project_model_is_flagged(tmp_path, monkeypatch):
    """Le candidat reste proposé, mais l'humain voit qu'il fait moins bien que le modèle du projet."""
    from mlflow_agent import orchestrator_adapter as adapter

    from common.schemas import Baseline
    from orchestrator.check_agent import prepare

    inp = prepare("agent2", CREDIT_RISK, tmp_path / "out")
    path = Path(inp.inputs["baseline"])
    b = Baseline.model_validate_json(path.read_text())
    b.observed_metrics["roc_auc"] = 0.99              # modèle du projet artificiellement excellent
    path.write_text(b.model_dump_json())
    result = adapter.run(inp)
    assert result.status.value == "NEEDS_APPROVAL"
    assert "moins bon que le modèle actuel du projet" in result.reasons[0]


def test_retraining_reads_agent4_parquet_and_compares_to_production(tmp_path):
    """Réentraînement : nouvelles données en parquet (Agent 4) ; le champion est jugé sur sa performance
    mesurée en production, pas sur son ancien jeu de test."""
    import pandas as pd
    from mlflow_agent.orchestrator_adapter import _dataset, _production_score

    from common.schemas import AgentInput, RetrainingRequest, write_output

    new = tmp_path / "new_data.parquet"
    # labels de production : réels 0.0 / 1.0 (table SQLite de la gateway) ; on reprend le type du projet
    pd.DataFrame({"amount": [1.0, 2.0], "target": [0.0, 1.0]}).to_parquet(new)
    wd = tmp_path / "agent2"
    wd.mkdir()
    req_inp = AgentInput(run_id="r", project_dir=str(CREDIT_RISK), report_path="", workdir=str(tmp_path),
                         inputs={})
    path = write_output(req_inp, "retraining_request", RetrainingRequest(
        model_uri="models:/credit_risk/1", reason="performance_drop", new_data_path=str(new),
        current_performance={"roc_auc": 0.64}))
    inp = AgentInput(run_id="r", project_dir=str(CREDIT_RISK), report_path="", workdir=str(wd),
                     inputs={"retraining_request": path})
    warnings = []
    spec = type("S", (), {"data_path": "data.csv", "target": "target"})()
    data = _dataset(inp, spec, warnings)
    assert pd.read_csv(data).to_dict("list") == {"amount": [1.0, 2.0], "target": [0, 1]}
    assert [line.split(",")[1] for line in data.read_text().splitlines()[1:]] == ["0", "1"]
    assert _production_score(inp, "roc_auc", warnings) == 0.64
    assert _production_score(inp, "f1", []) is None
