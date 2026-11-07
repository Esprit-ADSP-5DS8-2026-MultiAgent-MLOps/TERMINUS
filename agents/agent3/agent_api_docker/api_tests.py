"""Generation des tests de l'API a partir du rapport + runner (serveur local -> pytest).

Les tests generes ciblent une URL (variable BASE_URL) : ce sont les MEMES tests qui serviront
plus tard contre le conteneur Docker (il suffira de changer BASE_URL).

Le test de coherence (API == appel direct du modele) n'est execute que si RUN_DIR est fourni,
car il charge le modele dans le processus de test. Il detecte les "faux succes".
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .errors import classify, tail_relevant
from .schemas import AnalysisReport

TEST_FILENAME = "test_api_generated.py"

# --------------------------------------------------------------------------- generation

_TEMPLATE = '''\
"""Tests generes automatiquement a partir du rapport d'analyse. Ne pas editer."""
import importlib
import json
import os
import sys

import httpx
import pytest

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
RUN_DIR = os.environ.get("RUN_DIR")  # present seulement en mode local

EXAMPLE_INPUT = __EXAMPLE_INPUT__
EXAMPLE_OUTPUT = __EXAMPLE_OUTPUT__
FIELD_TYPES = __FIELD_TYPES__
REQUIRED_FIELDS = __REQUIRED__
OUTPUT_FIELDS = __OUTPUT_FIELDS__
DATA_FILE = __DATA_FILE__
PREDICT_MODULE = __PREDICT_MODULE__
LOADER = __LOADER__
PREDICT_FN = __PREDICT_FN__
MODEL_PATH = __MODEL_PATH__
CUSTOM_MODULE = __CUSTOM_MODULE__
SUPPORTS_PROBA = __SUPPORTS_PROBA__
N_CONSISTENCY_ROWS = 20

BAD_VALUES = {"int": "abc", "float": "abc", "bool": "maybe", "str": [1, 2]}


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("model_loaded") is True


def test_predict_example(client):
    r = client.post("/predict", json=EXAMPLE_INPUT)
    assert r.status_code == 200, r.text
    body = r.json()
    for name in OUTPUT_FIELDS:
        assert name in body, f"champ de sortie absent: {name}"
    if EXAMPLE_OUTPUT:
        expected = EXAMPLE_OUTPUT["prediction"]
        if isinstance(expected, float):   # regression : aller-retour JSON des entrees
            assert body["prediction"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
        else:
            assert body["prediction"] == expected
        if "proba" in EXAMPLE_OUTPUT and body.get("proba") is not None:
            assert body["proba"] == pytest.approx(EXAMPLE_OUTPUT["proba"], abs=0.02)


@pytest.mark.parametrize("missing", REQUIRED_FIELDS)
def test_missing_field_returns_422(client, missing):
    payload = {k: v for k, v in EXAMPLE_INPUT.items() if k != missing}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("name", list(FIELD_TYPES))
def test_wrong_type_returns_422(client, name):
    payload = dict(EXAMPLE_INPUT)
    payload[name] = BAD_VALUES[FIELD_TYPES[name]]
    assert client.post("/predict", json=payload).status_code == 422


def test_openapi_exposes_input_fields(client):
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    exposed = set()
    for s in schemas.values():
        exposed |= set(s.get("properties", {}))
    assert set(FIELD_TYPES) <= exposed


def test_api_matches_direct_model_call(client):
    """L'API doit donner EXACTEMENT les memes predictions que le modele appele directement."""
    if not RUN_DIR or not DATA_FILE:
        pytest.skip("RUN_DIR ou fichier de donnees absent")
    import pandas as pd

    sys.path.insert(0, RUN_DIR)
    if CUSTOM_MODULE:
        importlib.import_module(CUSTOM_MODULE)
    mod = importlib.import_module(PREDICT_MODULE)
    load = getattr(mod, LOADER) if LOADER else importlib.import_module("joblib").load
    model = load(os.path.join(RUN_DIR, MODEL_PATH))
    df = pd.read_csv(os.path.join(RUN_DIR, DATA_FILE))[list(FIELD_TYPES)].head(N_CONSISTENCY_ROWS)
    direct = getattr(mod, PREDICT_FN)(model, df)
    rows = json.loads(df.to_json(orient="records"))
    for i, row in enumerate(rows):
        r = client.post("/predict", json=row)
        assert r.status_code == 200, f"ligne {i}: {r.text}"
        if isinstance(direct[i], float):   # regression : aller-retour JSON des entrees
            assert r.json()["prediction"] == pytest.approx(direct[i], rel=1e-6, abs=1e-6), f"ligne {i}: API != modele"
        else:
            assert r.json()["prediction"] == direct[i], f"ligne {i}: API != modele"
        if SUPPORTS_PROBA and r.json().get("proba") is not None:
            p = float(model.predict_proba(df.iloc[[i]])[0, 1])
            assert r.json()["proba"] == pytest.approx(p, abs=1e-6), f"ligne {i}: proba differente"
'''


def _first_csv(report: AnalysisReport) -> Optional[str]:
    return next((f for f in report.project.file_tree if f.lower().endswith(".csv")), None)


def generate_tests(report: AnalysisReport) -> str:
    """Retourne le contenu du fichier pytest (sans appel LLM)."""
    ep = report.inference.entrypoint
    loader_fn = (ep.loader_function or "").split("(")[0].strip() or None   # None = joblib.load direct
    values = {
        "__EXAMPLE_INPUT__": report.io.example_input or {},
        "__EXAMPLE_OUTPUT__": report.io.example_output,
        "__FIELD_TYPES__": {f.name: f.type for f in report.io.input_schema},
        "__REQUIRED__": [f.name for f in report.io.input_schema if f.required],
        "__OUTPUT_FIELDS__": [f.name for f in report.io.output_schema if not f.optional],
        "__DATA_FILE__": _first_csv(report),
        "__PREDICT_MODULE__": ep.file.removesuffix(".py").replace("/", "."),
        "__LOADER__": loader_fn,
        "__PREDICT_FN__": ep.function,
        "__MODEL_PATH__": report.ml.model_file,
        "__CUSTOM_MODULE__": report.ml.custom_classes_module,
        "__SUPPORTS_PROBA__": report.inference.supports_proba,
    }
    code = _TEMPLATE
    for key, val in values.items():
        code = code.replace(key, repr(val))
    compile(code, TEST_FILENAME, "exec")  # garde-fou : le fichier genere doit etre valide
    return code


# --------------------------------------------------------------------------- preparation

def _argv(command: str, python_exe: str) -> list[str]:
    """Convertit 'python -c "..."' en argv portable (Windows compris)."""
    m = re.match(r'^python3?\s+-c\s+"(.*)"\s*$', command, re.S)
    if m:
        return [python_exe, "-c", m.group(1)]
    parts = shlex.split(command, posix=(os.name != "nt"))
    if parts and parts[0] in ("python", "python3"):
        parts[0] = python_exe
    return parts


def prepare_run_dir(project_dir: Path, generated_dir: Path, report: AnalysisReport, run_dir: Path,
                    python_exe: Optional[str] = None, train_timeout: int = 600) -> Path:
    """Assemble projet + fichiers generes dans run_dir, puis entraine le modele s'il est absent.

    run_dir peut etre generated_dir lui-meme : le dossier devient alors le dossier COMPLET (projet + fichiers generes
    + corrections), pret pour `docker build`. Les fichiers generes l'emportent sur les fichiers du projet de meme nom
    (api.py, Dockerfile...), y compris les correctifs deja rejoues avant l'appel.
    """
    py = python_exe or sys.executable
    run_dir = Path(run_dir)
    same = Path(generated_dir).resolve() == run_dir.resolve()
    keep = {f.name: f.read_bytes() for f in Path(generated_dir).iterdir() if f.is_file()} if same else {}
    shutil.copytree(project_dir, run_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("output", "expected_reports", ".git", ".venv", "venv", "__pycache__", "*.pyc"))
    if same:
        for name, data in keep.items():          # le projet a pu ecraser un fichier genere de meme nom : on le restaure
            (run_dir / name).write_bytes(data)
    else:
        for f in Path(generated_dir).iterdir():
            if f.is_file():
                shutil.copy2(f, run_dir / f.name)

    if not (run_dir / report.ml.model_file).exists():
        gen = report.ml.model_generation
        if not gen or not gen.command:
            raise RuntimeError(f"Modele absent ({report.ml.model_file}) et aucune commande d'entrainement dans le rapport")
        proc = subprocess.run(_argv(gen.command, py), cwd=run_dir, capture_output=True, text=True, timeout=train_timeout, env=config.clean_env())
        if proc.returncode != 0:
            raise RuntimeError("Entrainement du modele echoue:\n" + tail_relevant(proc.stderr or proc.stdout, 40))
    return run_dir


# --------------------------------------------------------------------------- runner

@dataclass
class TestRunResult:
    started: bool = False
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    tests: list[dict] = field(default_factory=list)
    server_logs: str = ""
    pytest_output: str = ""
    error_category: Optional[str] = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.started and self.failed == 0 and self.passed > 0

    def summary(self) -> str:
        """Texte compact a renvoyer au LLM pour la correction."""
        if not self.started:
            return f"[{self.error_category}] Le serveur n'a pas demarre:\n{tail_relevant(self.server_logs, 40)}"
        if self.ok:
            return f"{self.passed} tests OK ({self.skipped} ignores)"
        bad = [f"- {t['name']}: {t['message'][:300]}" for t in self.tests if t["status"] == "failed"]
        return f"{self.failed} test(s) en echec:\n" + "\n".join(bad)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(url: str, proc: subprocess.Popen, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _parse_junit(path: Path) -> list[dict]:
    tests = []
    for case in ET.parse(path).getroot().iter("testcase"):
        status, msg = "passed", ""
        for child in case:
            if child.tag in ("failure", "error"):
                status, msg = "failed", (child.get("message") or "")
            elif child.tag == "skipped":
                status, msg = "skipped", (child.get("message") or "")
        tests.append({"name": case.get("name"), "status": status, "message": msg})
    return tests


def _pytest_against(result: TestRunResult, run_dir: Path, base_url: str, py: str, with_run_dir: bool,
                    tests_timeout_s: int) -> None:
    """Lance le fichier de tests genere contre base_url et remplit `result`.

    with_run_dir=False (conteneur) : le test de coherence API == modele direct est ignore, car il charge le
    modele dans le processus de test (ce qui n'a de sens qu'en local).
    """
    junit = run_dir / "junit.xml"
    junit.unlink(missing_ok=True)
    extra = {"BASE_URL": base_url}
    if with_run_dir:
        extra["RUN_DIR"] = str(run_dir)
    proc = subprocess.run([py, "-m", "pytest", TEST_FILENAME, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                          cwd=run_dir, env=config.clean_env(extra), capture_output=True, text=True,
                          timeout=tests_timeout_s)
    result.pytest_output = proc.stdout[-3000:]
    if junit.exists():
        result.tests = _parse_junit(junit)
        for t in result.tests:
            setattr(result, t["status"], getattr(result, t["status"]) + 1)
    else:
        result.failed = 1
        result.tests = [{"name": "pytest", "status": "failed", "message": tail_relevant(proc.stdout + proc.stderr, 30)}]


def run_tests_against_url(report: AnalysisReport, run_dir: Path, base_url: str, python_exe: Optional[str] = None,
                          tests_timeout_s: int = 300) -> TestRunResult:
    """Memes tests generes, contre un serveur deja demarre (ex. le conteneur Docker). Ne demarre/n'arrete rien."""
    run_dir = Path(run_dir)
    t0 = time.time()
    (run_dir / TEST_FILENAME).write_text(generate_tests(report), encoding="utf-8")
    result = TestRunResult(started=True)
    _pytest_against(result, run_dir, base_url, python_exe or sys.executable, with_run_dir=False,
                    tests_timeout_s=tests_timeout_s)
    result.duration_s = round(time.time() - t0, 2)
    return result


def run_api_tests(report: AnalysisReport, run_dir: Path, python_exe: Optional[str] = None,
                  health_timeout_s: int = 60, tests_timeout_s: int = 300) -> TestRunResult:
    """Lance uvicorn sur un port libre, execute les tests generes, arrete toujours le serveur."""
    py = python_exe or sys.executable
    run_dir = Path(run_dir)
    result = TestRunResult()
    t0 = time.time()
    (run_dir / TEST_FILENAME).write_text(generate_tests(report), encoding="utf-8")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = run_dir / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen([py, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port)],
                                  cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=config.clean_env())
        try:
            result.started = _wait_health(base_url + "/health", server, health_timeout_s)
            if result.started:
                _pytest_against(result, run_dir, base_url, py, with_run_dir=True, tests_timeout_s=tests_timeout_s)
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
    result.server_logs = log_path.read_text(encoding="utf-8", errors="replace")
    if not result.started:
        result.error_category = classify(result.server_logs)[0]
    result.duration_s = round(time.time() - t0, 2)
    return result
