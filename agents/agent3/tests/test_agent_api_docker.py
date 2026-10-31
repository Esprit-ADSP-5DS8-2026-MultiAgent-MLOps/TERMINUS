import json
import shutil

import pytest

from agent_api_docker import config
from agent_api_docker.errors import classify
from agent_api_docker.generator import generate_all
from agent_api_docker.report_loader import NeedsUserInput, ReportError, load_report, resolve_report_path
from agent_api_docker.tools.publish import ApprovalRequired, docker_push


def test_report_found_in_eval_dir():
    assert resolve_report_path("credit_risk").name == "credit_risk.analysis_report.json"


@pytest.mark.parametrize("pid", ["credit_risk", "diabetes_progression", "ticket_classifier"])
def test_all_reference_reports_are_valid(pid):
    assert load_report(pid).project_name


def test_missing_report():
    with pytest.raises(ReportError):
        resolve_report_path("inconnu")


def test_generation_credit_risk():
    files = generate_all(load_report("credit_risk"))
    assert set(files) == {"api.py", "requirements.api.txt", "Dockerfile", ".dockerignore"}
    compile(files["api.py"], "api.py", "exec")  # syntaxe valide
    assert "import preprocessing" in files["api.py"]
    assert "0.0.0.0" in files["Dockerfile"] and "USER appuser" in files["Dockerfile"]
    assert "scikit-learn==1.8.0" in files["requirements.api.txt"]


def test_low_confidence_stops(tmp_path):
    data = json.loads((config.EVAL_REPORTS_DIR / "credit_risk.analysis_report.json").read_text())
    data["io"]["schema_confidence"] = "low"
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    with pytest.raises(NeedsUserInput):
        load_report("credit_risk", p)


def test_push_requires_approval():
    with pytest.raises(ApprovalRequired):
        docker_push("img", "1.0", None, "sha256:abc")


def test_classify():
    assert classify("ModuleNotFoundError: No module named x")[0] == "DEP_MISSING_PYTHON"
