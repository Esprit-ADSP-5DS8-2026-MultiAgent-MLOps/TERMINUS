"""L'adaptateur respecte le contrat de l'orchestrateur (vrai validateur, API testée en local, sans Docker ni LLM)."""

import json
from pathlib import Path

import pytest

from orchestrator.check_agent import check, prepare

ROOT = Path(__file__).resolve().parents[3]
CREDIT_RISK = ROOT / "eval" / "projects" / "credit_risk"
TARGET = "agent_api_docker.orchestrator_adapter:run"


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.chdir(ROOT)                      # le validateur est lancé avec « python -m validator »
    monkeypatch.setenv("AGENT3_DOCKER", "0")
    monkeypatch.setenv("AGENT3_NO_FIX", "1")


def test_project_model_is_served_when_mlflow_is_unavailable(tmp_path):
    """Entrées des stubs (MLflow fictif) : repli sur le modèle du projet, entraîné comme avant par l'agent."""
    rep = check("agent3", TARGET, CREDIT_RISK, tmp_path / "out")
    assert rep.ok, [t for ok, t in rep.checks if not ok]
    dep = json.loads((tmp_path / "out" / "agent3" / "deployment_result.json").read_text())
    assert dep["model_source"] == "project_artifact" and dep["endpoint"] is None
    assert all(c["passed"] for c in dep["checks"])


def test_mlflow_model_is_served(tmp_path, monkeypatch):
    """Un vrai modèle du registre MLflow : c'est lui qui est servi, avec les versions qui l'ont entraîné."""
    pytest.importorskip("mlflow")
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import sklearn
    from agent_api_docker.orchestrator_adapter import run
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from common.schemas import MlflowMetadata

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    df = pd.read_csv(CREDIT_RISK / "data.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    categorical = [c for c in X.columns if X[c].dtype == object]
    pipe = make_pipeline(make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categorical),
                                                 remainder="passthrough"),
                         LogisticRegression(max_iter=2000)).fit(X, y)
    mlflow.set_experiment("t")
    with mlflow.start_run():
        mlflow.sklearn.log_model(pipe, name="model", registered_model_name="credit_risk")

    inp = prepare("agent3", CREDIT_RISK, tmp_path / "out")
    path = Path(inp.inputs["mlflow_metadata"])
    meta = MlflowMetadata.model_validate_json(path.read_text())
    meta = meta.model_copy(update={"tracking_uri": uri, "model_uri": "models:/credit_risk/1", "model_version": "1"})
    path.write_text(meta.model_dump_json())
    result = run(inp)
    assert result.status.value == "WARNING" and result.can_proceed, result.reasons
    dep = json.loads(Path(result.outputs["deployment_result"]).read_text())
    assert dep["model_source"] == "mlflow" and dep["model_uri"] == "models:/credit_risk/1"
    reqs = (Path(inp.workdir) / "workspace" / "generated" / "requirements.api.txt").read_text()
    assert f"scikit-learn=={sklearn.__version__}" in reqs


def test_original_labels_are_restored():
    """Le modèle de l'Agent 2 prédit l'indice de la classe : mlflow_predict.py rend l'étiquette d'origine."""
    from agent_api_docker.orchestrator_adapter import _PREDICT_SRC

    ns: dict = {}
    exec(_PREDICT_SRC.format(classes=[1, 2]), ns)  # noqa: S102 - module généré par l'adaptateur lui-même

    class Model:
        def predict(self, df):
            import numpy as np
            return np.array([0, 1] * (len(df) // 2 or 1))[: len(df)]

    assert ns["predict"](Model(), [{"a": 1}, {"a": 2}]) == [1, 2]


def test_push_needs_the_flag_and_docker_hub_settings(monkeypatch):
    """Pas de publication sans AGENT3_PUSH=1, ni sans identifiants (vérifié avant le build)."""
    from agent_api_docker import config
    from agent_api_docker.orchestrator_adapter import _push_wanted

    warnings: list[str] = []
    monkeypatch.delenv("AGENT3_PUSH", raising=False)
    monkeypatch.setattr(config, "_DOTENV", {})
    assert not _push_wanted(warnings) and not warnings

    monkeypatch.setenv("AGENT3_PUSH", "1")
    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "")
    monkeypatch.setattr(config, "DOCKER_IMAGE_NAMESPACE", "")
    assert not _push_wanted(warnings) and "DOCKERHUB_TOKEN" in warnings[0]

    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "moi")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "jeton")
    monkeypatch.setattr(config, "DOCKER_IMAGE_NAMESPACE", "moi")
    assert _push_wanted([])


def test_terminal_asks_which_docker_hub_account(monkeypatch):
    """Entrée = compte du .env (sans redemander son token) ; un autre compte = son token, saisi sans écho."""
    from agent_api_docker import config
    from agent_api_docker.orchestrator_adapter import _credentials_prompt

    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "equipe")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "jeton-env")
    monkeypatch.setattr(config, "DOCKER_IMAGE_NAMESPACE_EXPLICIT", "")
    shown: list[str] = []

    def never(prompt):
        raise AssertionError("le token du .env ne doit pas être redemandé")

    ask = _credentials_prompt("credit_risk", input_fn=lambda p: "", secret_fn=never, out=shown.append)
    assert ask(None) == ("equipe", "jeton-env")
    assert any("docker.io/equipe/credit_risk" in line for line in shown)

    ask = _credentials_prompt("credit_risk", input_fn=lambda p: "syrine", secret_fn=lambda p: "jeton-syrine",
                              out=shown.append)
    assert ask(None) == ("syrine", "jeton-syrine")


def test_no_terminal_question_outside_the_command_line(monkeypatch):
    from agent_api_docker.orchestrator_adapter import _interactive

    monkeypatch.delenv("ORCH_INTERACTIVE", raising=False)
    assert not _interactive()       # --auto-approve, Streamlit, tests : identifiants du .env


def test_publish_choice_made_in_slack(tmp_path):
    """La gate champion validée dans Slack écrit runs/<id>/publish.json : l'Agent 3 publie ou non."""
    from agent_api_docker.orchestrator_adapter import _publish_choice

    from common.schemas import AgentInput

    wd = tmp_path / "run" / "agent3"
    wd.mkdir(parents=True)
    inp = AgentInput(run_id="r", project_dir=str(tmp_path), report_path="", workdir=str(wd))
    assert _publish_choice(inp) is None
    (wd.parent / "publish.json").write_text('{"push": false, "by": "aziz"}')
    assert _publish_choice(inp) is False
    (wd.parent / "publish.json").write_text('{"push": true, "by": "syrine"}')
    assert _publish_choice(inp) is True
