"""Jeu d'évaluation : annotations cohérentes, projets multi-frameworks, métriques."""
import json
import sys

import pytest
from helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import metrics  # noqa: E402
from code_map import check_for  # noqa: E402

from policy import load_contract  # noqa: E402

ANN = json.loads((ROOT / "eval" / "annotations.json").read_text())


def test_annotations_cover_baseline_variants_adversarial_and_new_frameworks():
    assert len([k for k in ANN if k[:2].isdigit()]) == 14 and {"A1_prompt_injection", "A2_obfuscated_system_call", "A3_fake_metrics"} <= set(ANN)
    fw = {v["framework"] for v in ANN.values()}
    assert {"scikit-learn", "pytorch", "tensorflow"} <= fw
    names = " ".join(ANN)
    for needle in ("nlp", "vision", "timeseries", "no_requirements", "pyproject", "notebook", "training_only"):
        assert needle in names, needle
    assert {v["split"] for v in ANN.values()} == {"dev", "holdout"}


@pytest.mark.parametrize("name", sorted(ANN))
def test_annotation_is_consistent_with_its_files_and_contract(name):
    a = ANN[name]
    assert (ROOT / "eval" / a["path"]).is_dir()
    c = load_contract(a["profile"])
    assert all(c.check(i) for i in a["expected_checks"]), "contrôle attendu absent du profil"
    assert a["expected_status"] in {"ACCEPTED", "REPAIR_REQUIRED", "REJECTED", "NEEDS_REVIEW"}
    if not a["expected_checks"]:
        assert a["expected_status"] == "ACCEPTED"


def test_every_baseline_code_maps_to_a_catalog_check():
    expected = json.loads((ROOT / "eval" / "expected_verdicts.json").read_text())
    for v in expected.values():
        for code in v["codes"]:
            cid = check_for(code)
            assert cid and load_contract("sklearn-tabular").check(cid), code


def test_multi_framework_projects_are_really_out_of_scope_or_conform():
    import ast
    for name in ("m01_pytorch_mnist", "m02_tensorflow_text", "m04_vision_torchvision"):
        src = (ROOT / "eval" / ANN[name]["path"] / "train.py").read_text()
        assert any(m in src for m in ("import torch", "import tensorflow")), name
    assert not (ROOT / "eval/projects/multi/m06_no_requirements/requirements.txt").exists()
    assert (ROOT / "eval/projects/multi/m07_pyproject_only/pyproject.toml").exists()
    nb = json.loads((ROOT / "eval/projects/multi/m08_notebook_only/analysis.ipynb").read_text())
    assert nb["cells"][0]["cell_type"] == "code"
    for p in (ROOT / "eval/projects/multi").rglob("*.py"):
        ast.parse(p.read_text())


def test_validator_metrics():
    rs = [{"name": "a", "expected_status": "ACCEPTED", "predicted_status": "ACCEPTED", "expected_checks": [], "predicted_checks": [], "split": "dev"},
          {"name": "b", "expected_status": "REPAIR_REQUIRED", "predicted_status": "ACCEPTED", "expected_checks": ["X", "Y"], "predicted_checks": ["X"], "split": "dev"},
          {"name": "c", "expected_status": "ACCEPTED", "predicted_status": "NEEDS_REVIEW", "expected_checks": [], "predicted_checks": ["Z"], "split": "holdout"},
          {"name": "d", "expected_status": "REJECTED", "predicted_status": "REJECTED", "expected_checks": ["S"], "predicted_checks": ["S"], "split": "holdout"}]
    m = metrics.score_validator(rs)["all"]
    assert m["issue_recall"] == pytest.approx(2 / 3, abs=1e-3) and m["issue_precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["false_accepts"] == ["b"] and m["false_accept_rate"] == pytest.approx(1 / 2) and m["false_alarms"] == ["c"] and m["false_alarm_rate"] == 0.5
    assert m["decision_reliability"] == 0.5 and m["needs_review_rate"] == 0.25
    assert metrics.score_validator(rs)["split:holdout"]["n"] == 2


def test_judge_and_repair_metrics():
    j = metrics.score_judge([{"name": "g1", "label": "good", "verdict": "ACCEPT"}, {"name": "g2", "label": "good", "verdict": "REVISE"},
                             {"name": "b1", "label": "bad", "verdict": "ACCEPT"}, {"name": "b2", "label": "bad", "verdict": "REJECT", "failed_closed": True}])
    assert j["false_accepts"] == ["b1"] and j["false_rejects"] == ["g2"] and j["accuracy"] == 0.5 and j["failed_closed"] == 1
    r = metrics.score_repair([{"name": "x", "repair_status": "REPAIRED", "attempts": 1, "revalidated_status": "ACCEPTED", "regressions": 0},
                              {"name": "y", "repair_status": "ESCALATED", "attempts": 2, "revalidated_status": None, "regressions": 1}])
    assert r["repair_rate"] == 0.5 and r["mean_attempts"] == 1.5 and r["regressions_introduced"] == 1 and r["escalated"] == ["y"]
