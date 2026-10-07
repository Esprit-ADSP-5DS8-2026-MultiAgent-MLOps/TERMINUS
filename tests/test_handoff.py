"""Passage de relais : empreinte, copie validée, sceau, vérification, cas bloqués."""
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from validator import validate
from validator.handoff import compute_project_hash, schema_json, seal, verify_handoff
from validator.models import ValidationReport

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "projects"


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    snap = tmp_path_factory.mktemp("snap")
    rep = validate(EVAL / "credit_risk", bandit=False, snapshot_dir=str(snap))
    return rep


def test_new_fields(good):
    assert good.schema_version == "1.0" and len(good.run_id) == 12
    assert re.fullmatch(r"[0-9a-f]{64}", good.project_hash) and good.hashed_files > 5
    assert good.created_at.endswith("+00:00") and len(good.report_sha256) == 64


def test_handoff_ready_content(good):
    h = good.handoff
    assert h.status in ("READY", "READY_WITH_WARNINGS") and h.can_proceed
    assert (h.task, h.target, h.primary_metric) == ("classification", "target", "roc_auc")
    assert len(h.feature_columns) == 20 and "target" not in h.feature_columns
    assert h.pinned_dependencies["scikit-learn"] == "1.8.0"
    assert h.observed_metrics["roc_auc"] > 0.75 and h.deterministic is True
    codes = {c.code for c in h.carry_forward}
    assert {"HANDOFF.IMBALANCED_CLASSES", "HANDOFF.SMALL_DATASET"} <= codes


def test_snapshot_matches_and_is_read_only(good):
    snap = Path(good.snapshot_path)
    assert snap.is_dir() and compute_project_hash(snap)[0] == good.project_hash
    assert all(not (p.stat().st_mode & 0o222) for p in snap.rglob("*") if p.is_file())


def test_verify_ok_with_project_dir_and_with_snapshot_only(good):
    assert verify_handoff(good, EVAL / "credit_risk").ok
    assert verify_handoff(good).ok  # utilise snapshot_path


def test_verify_detects_modified_project(good, tmp_path):
    copy = tmp_path / "p"
    shutil.copytree(EVAL / "credit_risk", copy, ignore=shutil.ignore_patterns("__pycache__", "output"))
    (copy / "train.py").write_text((copy / "train.py").read_text() + "\n# modifié\n")
    res = verify_handoff(good, copy)
    assert not res.ok and any("changé" in r for r in res.reasons)


def test_verify_detects_tampered_report(good, tmp_path):
    data = json.loads(good.model_dump_json())
    data["handoff"]["observed_metrics"]["roc_auc"] = 0.99
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    res = verify_handoff(p, EVAL / "credit_risk")
    assert not res.ok and any("sceau" in r for r in res.reasons)


def test_report_json_roundtrip_keeps_seal(good, tmp_path):
    p = tmp_path / "compatibility_report.json"
    p.write_text(good.model_dump_json(indent=2))
    assert verify_handoff(p, EVAL / "credit_risk").ok


def test_unsupported_schema_major(good):
    rep = good.model_copy(deep=True)
    rep.schema_version = "2.0"
    seal(rep)
    res = verify_handoff(rep, EVAL / "credit_risk")
    assert not res.ok and any("schéma" in r for r in res.reasons)


def test_rejected_is_blocked_and_never_snapshotted(tmp_path):
    rep = validate(EVAL / "variants/02_unpinned_deps", dynamic=False, bandit=False, snapshot_dir=str(tmp_path))
    assert rep.handoff.status == "BLOCKED" and not rep.handoff.can_proceed
    assert rep.snapshot_path is None and not list(tmp_path.iterdir())
    assert not verify_handoff(rep, EVAL / "variants/02_unpinned_deps").ok


def test_needs_confirmation_awaits_human():
    rep = validate(EVAL / "variants/09_no_manifest", bandit=False)
    assert rep.handoff.status == "AWAITING_CONFIRMATION" and rep.handoff.requires_human_confirmation
    assert not rep.handoff.can_proceed


def test_confirmation_is_a_trace_not_a_bypass():
    rep = validate(EVAL / "variants/09_no_manifest", bandit=False, confirmed_by="Sarra", confirm_note="ok")
    assert rep.confirmation.confirmed_by == "Sarra" and rep.confirmation.manifest_sha256 is None
    assert rep.verdict.value == "NEEDS_CONFIRMATION" and not rep.handoff.can_proceed


def test_confirmation_records_manifest_hash():
    rep = validate(EVAL / "credit_risk", dynamic=False, bandit=False, confirmed_by="Sarra")
    assert len(rep.confirmation.manifest_sha256) == 64


def test_no_dynamic_is_incomplete_even_if_verdict_accepted():
    rep = validate(EVAL / "credit_risk", dynamic=False, bandit=False)
    assert rep.verdict.value == "ACCEPTED" and rep.handoff.status == "INCOMPLETE"
    assert not rep.handoff.can_proceed and not verify_handoff(rep, EVAL / "credit_risk").ok


def test_hash_is_deterministic_and_run_id_unique():
    a = validate(EVAL / "credit_risk", dynamic=False, bandit=False)
    b = validate(EVAL / "credit_risk", dynamic=False, bandit=False)
    assert a.project_hash == b.project_hash and a.run_id != b.run_id


def test_committed_schema_file_is_up_to_date():
    f = ROOT / "docs" / "compatibility_report.schema.json"
    assert f.exists(), "générez-le : python -m validator.handoff schema docs/compatibility_report.schema.json"
    assert f.read_text(encoding="utf-8").strip() == schema_json().strip()
    assert "handoff" in json.loads(f.read_text())["properties"]
