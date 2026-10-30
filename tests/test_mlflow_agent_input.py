import copy
import json
import os

import pytest

from mlflow_agent.schemas import (
    InvalidReportError, NotReadyError, load_report, require_ready, resolve_paths,
)

SAMPLE = {
    "schema_version": "0.1",
    "project": {
        "project_path": "./project",
        "dataset_path": "./data/train.csv",
        "training_script": "./src/train.py",
    },
    "task": {
        "type": "classification",
        "target_column": "churn",
        "objective": "predict customer churn",
        "primary_metric": "f1",
        "secondary_metrics": ["precision", "recall"],
    },
    "dataset_analysis": {
        "rows": 10000,
        "columns": 20,
        "missing_values": {"total": 25, "columns_affected": ["age"]},
        "class_distribution": {"negative": 8000, "positive": 2000},
        "data_leakage_risks": [],
    },
    "existing_implementation": {
        "training_code_available": True,
        "preprocessing_available": True,
        "existing_models": ["RandomForestClassifier"],
    },
    "constraints": {"max_experiment_runs": 30, "max_runtime_minutes": 60},
    "status": "ready_for_experimentation",
}


def make_workspace(base):
    (base / "project").mkdir(parents=True)
    (base / "data").mkdir()
    (base / "data" / "train.csv").write_text("a,churn\n1,0\n")
    (base / "src").mkdir()
    (base / "src" / "train.py").write_text("")
    return base


def write_report(tmp_path, data):
    p = tmp_path / "report.json"
    p.write_text(json.dumps(data))
    return p


def mutated(fn):
    d = copy.deepcopy(SAMPLE)
    fn(d)
    return d


def test_valid_report_loads_and_resolves(tmp_path):
    ws = make_workspace(tmp_path / "ws")
    report = load_report(write_report(tmp_path, SAMPLE))
    require_ready(report)
    paths = resolve_paths(report, ws)
    assert paths.dataset == (ws / "data" / "train.csv").resolve()
    assert report.task.primary_metric == "f1"


def test_unknown_field_rejected(tmp_path):
    bad = mutated(lambda d: d.update(shell_command="rm -rf /"))
    with pytest.raises(InvalidReportError):
        load_report(write_report(tmp_path, bad))


def test_metric_must_match_task_type(tmp_path):
    bad = mutated(lambda d: d["task"].update(primary_metric="rmse"))
    with pytest.raises(InvalidReportError):
        load_report(write_report(tmp_path, bad))


def test_classification_requires_class_distribution(tmp_path):
    bad = mutated(lambda d: d["dataset_analysis"].pop("class_distribution"))
    with pytest.raises(InvalidReportError):
        load_report(write_report(tmp_path, bad))


def test_constraints_above_hard_limit_rejected(tmp_path):
    bad = mutated(lambda d: d["constraints"].update(max_experiment_runs=10_000))
    with pytest.raises(InvalidReportError):
        load_report(write_report(tmp_path, bad))


def test_not_ready_status(tmp_path):
    data = mutated(lambda d: d.update(status="blocked"))
    report = load_report(write_report(tmp_path, data))
    with pytest.raises(NotReadyError):
        require_ready(report)


def test_invalid_json_and_missing_file(tmp_path):
    p = tmp_path / "report.json"
    p.write_text("{not json")
    with pytest.raises(InvalidReportError):
        load_report(p)
    with pytest.raises(InvalidReportError):
        load_report(tmp_path / "nope.json")


@pytest.mark.parametrize("evil", ["../../etc/passwd", "/etc/passwd"])
def test_path_escape_rejected(tmp_path, evil):
    ws = make_workspace(tmp_path / "ws")
    bad = mutated(lambda d: d["project"].update(dataset_path=evil))
    report = load_report(write_report(tmp_path, bad))
    with pytest.raises(InvalidReportError):
        resolve_paths(report, ws)


def test_symlink_escape_rejected(tmp_path):
    ws = make_workspace(tmp_path / "ws")
    secret = tmp_path / "secret.csv"
    secret.write_text("secret")
    os.symlink(secret, ws / "data" / "link.csv")
    bad = mutated(lambda d: d["project"].update(dataset_path="./data/link.csv"))
    report = load_report(write_report(tmp_path, bad))
    with pytest.raises(InvalidReportError):
        resolve_paths(report, ws)


def test_missing_dataset_rejected(tmp_path):
    ws = make_workspace(tmp_path / "ws")
    (ws / "data" / "train.csv").unlink()
    report = load_report(write_report(tmp_path, SAMPLE))
    with pytest.raises(InvalidReportError):
        resolve_paths(report, ws)