"""Tests de validate_api.py et api_tests.py, dont des mutations qui cassent volontairement l'API."""
import shutil
from pathlib import Path

import pytest

from agent_api_docker import config
from agent_api_docker.api_tests import generate_tests, prepare_run_dir, run_api_tests
from agent_api_docker.generator import generate_all
from agent_api_docker.report_loader import load_report
from agent_api_docker.tools.files import write_file
from agent_api_docker.validate_api import validate_api

PROJECT = config.EVAL_PROJECTS_DIR / "credit_risk"
needs_project = pytest.mark.skipif(not PROJECT.is_dir(), reason="eval/projects/credit_risk absent")


@pytest.fixture(scope="module")
def report():
    return load_report("credit_risk")


def test_generated_tests_are_valid_python(report):
    code = generate_tests(report)
    compile(code, "t.py", "exec")
    assert "test_api_matches_direct_model_call" in code and "'age'" in code


@pytest.fixture(scope="module")
def base_run(report, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    gen = tmp / "generated"
    gen.mkdir()
    for k, v in generate_all(report).items():
        write_file(gen, k, v)
    return prepare_run_dir(PROJECT, gen, report, tmp / "base")


def _mutate(base_run, tmp_path, old, new):
    dst = tmp_path / "mut"
    shutil.copytree(base_run, dst)
    f = dst / "api.py"
    text = f.read_text()
    assert old in text, f"motif introuvable: {old}"
    f.write_text(text.replace(old, new))
    return dst


@needs_project
def test_valid_api_passes_everything(report, base_run):
    v = validate_api(report, base_run)
    assert v.ok, v.summary()
    t = run_api_tests(report, base_run)
    assert t.ok, t.summary()
    assert t.skipped == 0  # le test de coherence API/modele a bien tourne


def test_syntax_error_detected(report, tmp_path):
    (tmp_path / "api.py").write_text("def broken(:\n")
    v = validate_api(report, tmp_path)
    assert not v.ok and v.failures[0].category == "API_SYNTAX"


@needs_project
def test_bad_import_detected(report, base_run, tmp_path):
    run = _mutate(base_run, tmp_path, "from predict import", "from predict_oops import")
    v = validate_api(report, run)
    assert not v.ok and v.failures[0].name == "import_app" and v.failures[0].category == "API_IMPORT"


@needs_project
def test_schema_mismatch_detected(report, base_run, tmp_path):
    run = _mutate(base_run, tmp_path, "    age: int", "    age_years: int")
    v = validate_api(report, run)
    assert any(c.name == "contract_input" and not c.passed for c in v.checks)


@needs_project
def test_wrong_model_path_server_does_not_start(report, base_run, tmp_path):
    run = _mutate(base_run, tmp_path, "output/model.joblib", "output/missing.joblib")
    t = run_api_tests(report, run, health_timeout_s=15)
    assert not t.started and not t.ok and t.error_category == "API_MODEL_LOAD"


@needs_project
def test_false_success_caught_by_tests(report, base_run, tmp_path):
    """L'API demarre et passe la validation statique, mais renvoie de mauvaises predictions."""
    run = _mutate(base_run, tmp_path, 'out = {"prediction": int(pred)}', 'out = {"prediction": 1 - int(pred)}')
    assert validate_api(report, run).ok            # invisible statiquement
    t = run_api_tests(report, run)
    failed = {x["name"] for x in t.tests if x["status"] == "failed"}
    assert not t.ok and {"test_predict_example", "test_api_matches_direct_model_call"} <= failed
