"""Service Slack de l'orchestrateur, sans Slack : un faux client enregistre les messages (stubs, vrai validateur)."""

import json
from pathlib import Path

import pytest

from orchestrator.graph import Run, run_pipeline
from orchestrator.slack import (
    APPROVE,
    PUBLISH,
    REJECT,
    OrchestratorSlack,
    Settings,
    project_of,
)
from orchestrator.stubs.agents import SIMULATE_FILE

ROOT = Path(__file__).resolve().parents[2]
CREDIT_RISK = str(ROOT / "eval" / "projects" / "credit_risk")


class FakeSlack:
    def __init__(self):
        self.posts, self.updates = [], []

    def chat_postMessage(self, **kw):
        self.posts.append(kw)
        return {"ts": f"{len(self.posts)}.0"}

    def chat_update(self, **kw):
        self.updates.append(kw)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_SIMULATE_DRIFT", raising=False)
    monkeypatch.setenv("AGENT3_PUSH", "1")
    monkeypatch.setenv("DOCKERHUB_USERNAME", "demo")
    monkeypatch.setenv("MONITORING_RUNS_DIR", str(tmp_path / "agent4"))
    settings = Settings({"ORCH_SLACK_CHANNEL": "C1", "ORCH_SLACK_APPROVERS": "UALL,UAZIZ,USYRINE",
                         "ORCH_SLACK_APPROVERS_CHAMPION_REPLACEMENT": "UAZIZ,USYRINE",
                         "ORCH_SLACK_APPROVERS_DOCKER_PUSH": "USYRINE"})
    svc = OrchestratorSlack(FakeSlack(), settings, str(tmp_path / "runs"), sync=True)
    svc.started_at = 0
    return svc


def _buttons(post):
    return [e for b in post.get("blocks", []) if b["type"] == "actions" for e in b["elements"]]


def test_gate_is_posted_once_and_answered_by_an_allowed_person(service):
    run = Run.start(CREDIT_RISK, service.runs_dir)                      # s'arrête à la gate champion
    service.tick()
    service.tick()                                                      # pas de doublon
    start, gate = service.client.posts
    assert "Pipeline sur" in start["text"] and gate["thread_ts"] == "1.0" and gate["reply_broadcast"]
    ids = [b["action_id"] for b in _buttons(gate)]
    assert ids == [PUBLISH, APPROVE, REJECT]                            # publication proposée (AGENT3_PUSH=1)
    text = json.dumps(gate["blocks"])
    assert "docker.io/demo/" in text and "<@UAZIZ>" in text and "UALL" not in text
    value = _buttons(gate)[0]["value"]

    assert "pas dans la liste" in service.on_click(APPROVE, value, "UALL", "all")
    assert "Docker Hub" in service.on_click(PUBLISH, value, "UAZIZ", "aziz")   # champion oui, publication non
    msg = service.on_click(PUBLISH, value, "USYRINE", "syrine", channel="C1", ts="2.0")
    assert "publication" in msg
    assert json.loads((run.run_dir / "publish.json").read_text())["push"] is True
    assert not _buttons(service.client.updates[-1])                     # boutons retirés du message
    assert Run.load(run.run_id, service.runs_dir).status == "DONE"
    assert "terminé" in service.client.posts[-1]["text"]
    assert "déjà été traitée" in service.on_click(APPROVE, value, "USYRINE", "syrine")
    decision = Run.load(run.run_id, service.runs_dir).state["decisions"][-1]
    assert decision["by"] == "syrine (Slack)" and decision["decision"] == "approve"


def test_rejecting_the_champion_ends_the_run(service):
    run = Run.start(CREDIT_RISK, service.runs_dir)
    service.tick()
    value = _buttons(service.client.posts[-1])[0]["value"]
    service.on_click(REJECT, value, "UAZIZ", "aziz")
    assert Run.load(run.run_id, service.runs_dir).status == "REJECTED"
    assert "refusé" in service.client.posts[-1]["text"]
    assert not (run.run_dir / "publish.json").exists()


def test_agent4_request_relaunches_monitoring_and_asks_in_slack(service, tmp_path):
    s = run_pipeline(CREDIT_RISK, lambda q: {"decision": "approve", "by": "t", "comment": ""},
                     runs_dir=service.runs_dir)
    run = Run.load(s["run_id"], service.runs_dir)
    (run.run_dir / SIMULATE_FILE).write_text(json.dumps({"drift": True, "failure": None}))   # le stub demandera
    served = json.loads(Path(run.state["artifacts"]["deployment_result"]).read_text())["model_uri"]
    _gateway_request(tmp_path, project_of(run), served)
    service.tick()
    texts = [p["text"] for p in service.client.posts]
    assert any("demande un réentraînement" in t for t in texts)
    assert Run.load(run.run_id, service.runs_dir).question["gate"] == "retraining"
    assert "Réentraînement demandé" in service.client.posts[-1]["text"]
    n = len(service.client.posts)
    service.tick()                                                      # même demande : rien de plus
    assert len(service.client.posts) == n


def _gateway_request(tmp_path, folder, model_uri, project=None):
    """Ce qu'écrit la gateway de l'Agent 4 : monitoring.json du projet et actions/retraining_request.json."""
    d = tmp_path / "agent4" / folder
    (d / "actions").mkdir(parents=True)
    (d / "monitoring.json").write_text(json.dumps({"project": project or folder}))
    (d / "actions" / "retraining_request.json").write_text(json.dumps({"model_uri": model_uri}))


def test_stale_or_archived_requests_never_retrain(service, tmp_path):
    s = run_pipeline(CREDIT_RISK, lambda q: {"decision": "approve", "by": "t", "comment": ""},
                     runs_dir=service.runs_dir)
    run = Run.load(s["run_id"], service.runs_dir)
    _gateway_request(tmp_path, project_of(run), "unknown")                       # ancienne demande
    _gateway_request(tmp_path, project_of(run) + ".old", "unknown", project_of(run))   # archive
    service.tick()
    texts = [p["text"] for p in service.client.posts]
    assert sum("ignorée" in t for t in texts) == 1 and not any(".old" in t for t in texts)
    assert Run.load(run.run_id, service.runs_dir).status == "DONE"


def test_request_without_deployed_run_is_reported(service, tmp_path):
    _gateway_request(tmp_path, "inconnu", "models:/inconnu/1")
    service.tick()
    service.tick()
    assert sum("aucun run déployé" in p["text"] for p in service.client.posts) == 1


def test_publish_button_only_when_publication_is_enabled(service, monkeypatch):
    monkeypatch.setenv("AGENT3_PUSH", "0")
    Run.start(CREDIT_RISK, service.runs_dir)
    service.tick()
    assert [b["action_id"] for b in _buttons(service.client.posts[-1])] == [APPROVE, REJECT]


def test_approvers_fall_back_to_the_default_list_then_the_channel():
    s = Settings({"ORCH_SLACK_APPROVERS": "U1", "ORCH_SLACK_APPROVERS_RETRAINING": "U2"})
    assert s.can("retraining", "U2") and not s.can("retraining", "U1")
    assert s.can("failure_review", "U1") and not s.can("failure_review", "U9")
    assert Settings({}).can("champion_replacement", "anyone")
