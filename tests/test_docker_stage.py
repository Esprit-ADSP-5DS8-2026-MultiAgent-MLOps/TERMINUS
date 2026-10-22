"""Etape Docker : tests sans Docker reel ni reseau.

FakeDocker remplace la CLI : `build` rejoue des resultats scriptes ; `run` lance un VRAI uvicorn sur l'api.py generee
(donc /health et les tests pytest generes tournent pour de bon). Faux LLM pour les corrections.
"""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from agent_api_docker import config
from agent_api_docker.api_tests import prepare_run_dir
from agent_api_docker.docker_stage import ApprovalRequest, publish_image, run_docker_stage
from agent_api_docker.generator import generate_all
from agent_api_docker.llm import LLMResponse
from agent_api_docker.schemas import AnalysisReport
from agent_api_docker.tools.docker import DockerCLI
from agent_api_docker.tools.publish import ApprovalRequired, PushError, docker_push
from agent_api_docker.tools.files import write_file

IMG_ID = "sha256:" + "ab" * 32

PREDICT = '''import joblib
import pandas as pd


def load_model(path):
    return joblib.load(path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else data
    return [int(x) for x in model.predict(df)]
'''
TRAIN = '''import joblib
from sklearn.linear_model import LogisticRegression

X = [[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [0.9, 1.2]]
joblib.dump(LogisticRegression().fit(X, [0, 1, 0, 1]), "model.joblib")
'''
REPORT = {
    "schema_version": "1", "project_name": "Mini Risk",
    "python": {"version": "3.12"},
    "dependencies": {"packages": [{"name": "scikit-learn", "version": "1.5.0"}, {"name": "pandas"}, {"name": "joblib"}]},
    "ml": {"framework": "sklearn", "model_file": "model.joblib", "model_exists_in_project": False, "loader": "joblib",
           "model_generation": {"required_before_serving": True, "command": "python train.py"}},
    "inference": {"entrypoint": {"file": "predict.py", "function": "predict", "loader_function": "load_model(path)"},
                  "supports_proba": True, "preprocessing_location": "inside_model"},
    "io": {"input_type": "tabular", "input_schema": [{"name": "x1", "type": "float"}, {"name": "x2", "type": "float"}],
           "output_schema": [{"name": "prediction", "type": "int"}, {"name": "proba", "type": "float", "optional": True}],
           "example_input": {"x1": 0.1, "x2": 0.0}, "example_output": {"prediction": 0}},
    "resources": {"gpu_required": "no"},
    "existing_app": {"port": 8000},
    "project": {"runtime_files": ["predict.py"]},
}


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, messages, tools):
        self.calls += 1
        return LLMResponse(text=json.dumps(self.replies.pop(0)), model="fake")


class FakeDocker(DockerCLI):
    """build_results : liste de (code, logs) ; run_mode : 'serve' (vrai uvicorn) | 'crash'."""

    def __init__(self, build_results, run_mode="serve", available=(True, "27.0"), image_id=IMG_ID):
        self.build_results, self.run_mode, self._available, self._id = list(build_results), run_mode, available, image_id
        self.builds, self.procs, self.cleaned, self.calls, self.stdin_seen = 0, {}, [], [], []
        self.context = None

    def available(self):
        return self._available

    def build(self, context_dir, tag, timeout_s=600, dockerfile="Dockerfile", on_line=None):
        self.builds += 1
        self.context = Path(context_dir)
        code, logs = self.build_results.pop(0)
        for line in logs.splitlines():
            on_line and on_line(line)
        return {"exit_code": code, "logs": logs}

    def image_id(self, ref):
        return self._id

    def image_size_mb(self, ref):
        return 123.4

    def run(self, image, port, env=None, name=None):
        if self.run_mode == "crash":
            return {"exit_code": 0, "logs": "cid1", "container_id": "cid1", "host_port": 9}
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            hp = s.getsockname()[1]
        p = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(hp)],
                             cwd=self.context, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs["cid1"] = p
        return {"exit_code": 0, "logs": "cid1", "container_id": "cid1", "host_port": hp}

    def state(self, cid):
        if self.run_mode == "crash":
            return {"status": "exited", "exit_code": 1, "health": ""}
        p = self.procs.get(cid)
        return {"status": "running" if p and p.poll() is None else "exited", "exit_code": 1, "health": ""}

    def logs(self, cid, tail=200):
        if self.run_mode == "crash":
            return "Traceback (most recent call last):\nNotADirectoryError: [Errno 20] Not a directory: 'model/regressor.joblib'"
        return "uvicorn logs"

    def cleanup(self, cid):
        self.cleaned.append(cid)
        p = self.procs.pop(cid, None)
        if p:
            p.terminate()
            p.wait(timeout=5)

    # --- Docker Hub : on enregistre les appels (le token ne doit apparaitre que sur stdin)
    def _exec(self, args, timeout=60, stdin=None, on_line=None):
        self.calls.append(args)
        if stdin is not None:
            self.stdin_seen.append(stdin)
        if args[0] == "push":
            return 0, "latest: digest: sha256:" + "cd" * 32 + " size: 1234"
        return 0, ""


@pytest.fixture()
def env(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "predict.py").write_text(PREDICT)
    (project / "train.py").write_text(TRAIN)
    report = AnalysisReport.model_validate(REPORT)
    ws = tmp_path / "ws"
    generated = ws / "generated"
    generated.mkdir(parents=True)
    for rel, content in generate_all(report).items():
        write_file(generated, rel, content)
    run_dir = prepare_run_dir(project, generated, report, ws / "run")
    return report, generated, run_dir, ws


def stage(env, docker, llm=None, **kw):
    report, generated, run_dir, ws = env
    msgs = []
    res = run_docker_stage(report, generated, run_dir, ws, step=msgs.append, on_step=msgs.append, on_detail=msgs.append,
                           llm=llm, docker=docker, sleep=lambda s: None, **kw)
    return res, msgs


def test_happy_path_build_run_tests(env):
    d = FakeDocker([(0, "#1 [internal] load build definition\n#2 [builder 3/6] RUN pip install\n")])
    res, msgs = stage(env, d)
    assert res.ok and res.stage == "done" and res.image_id == IMG_ID
    assert res.tests.passed > 0 and res.tests.failed == 0
    assert d.cleaned == ["cid1"]                                   # conteneur toujours supprime
    assert any("[builder 3/6]" in m for m in msgs)                 # progression du build visible


def test_build_failure_is_fixed_by_llm_then_succeeds(env):
    report, generated, run_dir, ws = env
    err = "ERROR: Could not find a version that satisfies the requirement scikit-learnn==9\nERROR: No matching distribution found for scikit-learnn==9"
    d = FakeDocker([(1, err), (0, "ok")])
    llm = FakeLLM({"diagnosis": "nom de paquet errone", "target": "docker",
                   "patches": [{"path": "requirements.api.txt", "old": "scikit-learn==1.5.0", "new": "scikit-learn==1.5.0  # ok"}]})
    fixes = []
    res, _ = stage(env, d, llm, fixes=fixes)
    assert res.ok and d.builds == 2 and llm.calls == 1
    assert fixes[0].stage == "build" and fixes[0].category == "DEP_MISSING_PYTHON" and fixes[0].applied == 1
    assert "# ok" in (run_dir / "requirements.api.txt").read_text()   # patch republie dans le contexte de build


def test_transient_network_error_retries_without_llm(env):
    d = FakeDocker([(1, "failed to do request: TLS handshake timeout"), (0, "ok")])
    llm = FakeLLM()
    res, msgs = stage(env, d, llm)
    assert res.ok and d.builds == 2 and llm.calls == 0
    assert any("nouvel essai" in m for m in msgs)


def test_docker_daemon_down_stops_without_llm(env):
    d = FakeDocker([], available=(False, "Le daemon Docker ne repond pas : lancer Docker Desktop"))
    llm = FakeLLM()
    res, _ = stage(env, d, llm)
    assert not res.ok and res.category == "DOCKER_DAEMON" and res.stage == "preflight" and llm.calls == 0


def test_env_failure_is_not_sent_to_llm(env):
    d = FakeDocker([(1, "toomanyrequests: You have reached your pull rate limit")])
    llm = FakeLLM()
    res, msgs = stage(env, d, llm)
    assert not res.ok and res.category == "REGISTRY_RATE_LIMIT" and llm.calls == 0 and d.builds == 1


def test_container_crash_is_fixed(env):
    d = FakeDocker([(0, "ok"), (0, "ok")], run_mode="crash")
    llm = FakeLLM({"diagnosis": "chemin du modele", "target": "docker", "patches": []})
    res, _ = stage(env, d, llm, max_fixes=1)
    assert not res.ok and res.stage == "health"
    assert res.category == "API_MODEL_LOAD"
    assert llm.calls == 1 and d.cleaned == ["cid1"]


def test_fix_budget_and_no_fix_flag(env):
    d = FakeDocker([(1, "ERROR: failed to solve: boom")])
    res, _ = stage(env, d, FakeLLM(), fix=False)
    assert not res.ok and d.builds == 1


# --------------------------------------------------------------------------- push

def _res(env):
    report = env[0]
    from agent_api_docker.docker_stage import DockerResult
    return report, DockerResult(ok=True, image="mini-risk:agent-build", image_id=IMG_ID, size_mb=1.0)


def test_push_without_approval_never_touches_registry(env):
    report, res = _res(env)
    d = FakeDocker([])
    out = publish_image(res, report, None, lambda m: None, env[3], docker=d, namespace="sarra")
    assert out.declined and not out.pushed and d.calls == []


def test_push_declined_by_user(env):
    report, res = _res(env)
    d = FakeDocker([])
    out = publish_image(res, report, lambda req: False, lambda m: None, env[3], docker=d, namespace="sarra")
    assert out.declined and d.calls == []


def test_push_approved_logs_in_via_stdin_and_pushes_all_tags(env, monkeypatch):
    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "sarra")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "s3cret-token")
    report, res = _res(env)
    d = FakeDocker([])
    seen = []
    out = publish_image(res, report, lambda req: seen.append(req) or True, lambda m: None, env[3], docker=d, namespace="sarra")
    assert out.pushed and out.digest == "sha256:" + "cd" * 32
    assert out.repository == "sarra/mini-risk" and out.tags[1] == "latest"
    assert seen[0].image_id == IMG_ID
    assert d.stdin_seen == ["s3cret-token\n"]                                   # token sur stdin...
    assert not any("s3cret-token" in a for call in d.calls for a in call)       # ...jamais en argument
    assert d.calls[-1] == ["logout"]
    assert [c[0] for c in d.calls].count("push") == 2


def test_push_refused_when_image_changed_since_approval():
    d = FakeDocker([], image_id="sha256:" + "ff" * 32)
    with pytest.raises(ApprovalRequired):
        docker_push("sarra/x", "latest", IMG_ID, IMG_ID, source_ref="x:local", docker=d, username="u", token="t")
    assert d.calls == []


def test_push_refused_without_or_with_wrong_digest():
    d = FakeDocker([])
    with pytest.raises(ApprovalRequired):
        docker_push("sarra/x", "latest", None, IMG_ID, source_ref="x", docker=d)
    with pytest.raises(ApprovalRequired):
        docker_push("sarra/x", "latest", "sha256:other", IMG_ID, source_ref="x", docker=d)


def test_push_missing_credentials(env, monkeypatch):
    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "")
    report, res = _res(env)
    out = publish_image(res, report, lambda r: True, lambda m: None, env[3], docker=FakeDocker([]), namespace="sarra")
    assert not out.pushed and "DOCKERHUB_TOKEN" in out.error


def test_invalid_image_name_rejected():
    with pytest.raises(ValueError):
        docker_push("-rf/x", "latest", IMG_ID, IMG_ID, source_ref="x", docker=FakeDocker([]), username="u", token="t")


# --------------------------------------------------------------------------- pipeline complet

def _pipeline(tmp_path, **kw):
    from agent_api_docker.loop import run_pipeline
    project = tmp_path / "p"
    project.mkdir()
    (project / "predict.py").write_text(PREDICT)
    (project / "train.py").write_text(TRAIN)
    rpt = tmp_path / "report.json"
    rpt.write_text(json.dumps(REPORT))
    logs = []
    res = run_pipeline("mini", str(rpt), project, workspace=tmp_path / "ws", agent="never",
                       on_step=logs.append, on_detail=logs.append, **kw)
    return res, logs


def test_pipeline_docker_and_push_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "sarra")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "tok")
    d = FakeDocker([(0, "ok")])
    res, logs = _pipeline(tmp_path, docker=True, push=True, approve_push=lambda r: True, docker_cli=d, namespace="sarra")
    assert res.ok and res.docker.ok and res.push.pushed
    steps = [m for m in logs if m.startswith("[") and "/8]" in m]
    assert len(steps) == 8                                  # 5 etapes API + build + conteneur + push
    data = json.loads((tmp_path / "ws" / "result.json").read_text())
    assert data["docker"]["ok"] and data["push"]["pushed"]
    traces = [json.loads(l)["step"] for l in (tmp_path / "ws" / "traces.jsonl").read_text().splitlines()]
    assert "docker" in traces and "push" in traces


def test_pipeline_push_declined_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "sarra")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "tok")
    d = FakeDocker([(0, "ok")])
    res, _ = _pipeline(tmp_path, push=True, approve_push=lambda r: False, docker_cli=d, namespace="sarra")
    assert res.ok and res.push.declined and not res.push.pushed
    assert not any(c[0] in ("login", "push") for c in d.calls)


def test_pipeline_without_docker_flag_never_calls_docker(tmp_path):
    d = FakeDocker([])
    res, _ = _pipeline(tmp_path, docker_cli=d)
    assert res.ok and res.docker is None and d.builds == 0


def test_pipeline_docker_failure_makes_result_fail(tmp_path):
    d = FakeDocker([(1, "ERROR: failed to solve: boom")])
    res, _ = _pipeline(tmp_path, docker=True, docker_cli=d, fix=False)
    assert res.api_ok and not res.ok and res.docker.stage == "build"


# --------------------------------------------------------------------------- templates Docker

def test_dockerfile_pythonpath_uses_major_minor_only():
    import copy
    R = copy.deepcopy(REPORT)
    R["python"]["version"] = "3.11.4"
    df = generate_all(AnalysisReport.model_validate(R))["Dockerfile"]
    assert "FROM python:3.11.4-slim" in df
    assert "/install/lib/python3.11/site-packages" in df and "python3.11.4/site-packages" not in df


def test_env_vars_are_quoted_and_never_none():
    import copy
    R = copy.deepcopy(REPORT)
    R["existing_app"]["env_vars"] = [{"name": "A", "default": None}, {"name": "B", "default": 'x "y" \\z'}]
    R["existing_app"]["external_services"] = ["redis"]
    files = generate_all(AnalysisReport.model_validate(R))
    assert 'ENV A=""' in files["Dockerfile"] and 'ENV B="x \\"y\\" \\\\z"' in files["Dockerfile"]
    assert "None" not in files["Dockerfile"] and "None" not in files["docker-compose.yml"]
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(files["docker-compose.yml"])["services"]["api"]["environment"]["A"] == ""


def test_push_without_settings_fails_before_any_build(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCKERHUB_USERNAME", "")
    monkeypatch.setattr(config, "DOCKERHUB_TOKEN", "")
    monkeypatch.setattr(config, "DOCKER_IMAGE_NAMESPACE", "")
    d = FakeDocker([])
    with pytest.raises(RuntimeError, match="DOCKERHUB_TOKEN"):
        _pipeline(tmp_path, push=True, approve_push=lambda r: True, docker_cli=d)
    assert d.builds == 0
