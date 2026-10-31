"""Tests for report-only ML review and deterministic fixes."""

from pathlib import Path

from agents.agent1.nodes.ml_fix import ml_fix
from agents.agent1.nodes.ml_review import ml_review, run_rules


def _source() -> str:
    """Return the intentionally leaky fixture source."""
    return Path("agents/agent1/tests/fixtures/leaky_project/train.py").read_text(encoding="utf-8")


def test_each_fixture_risk_rule_is_reported() -> None:
    """The fixture reports leakage, reproducibility, split, imbalance, and CV risks."""
    findings = run_rules(_source(), {"file": "train.py", "problem_type": "classification", "imbalanced": True})
    rules = {finding["rule"] for finding in findings}
    assert {"DATA_LEAKAGE", "NO_RANDOM_STATE", "NO_STRATIFY", "IMBALANCE_NOT_HANDLED", "NO_CROSS_VALIDATION"} <= rules


def test_ml_review_is_report_only(tmp_path: Path) -> None:
    """Review produces findings without changing source files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = _source()
    (workspace / "train.py").write_text(source, encoding="utf-8")
    before = (workspace / "train.py").read_bytes()
    result = ml_review({"workspace_path": str(workspace), "csv_path": "agents/agent1/tests/fixtures/leaky_project/data.csv", "problem_type": "classification", "profile": {}})
    assert result["ml_review"]["findings"]
    assert (workspace / "train.py").read_bytes() == before


def test_deterministic_fixes_apply_and_missing_key_skips_llm(tmp_path: Path) -> None:
    """Deterministic fixes apply even when the ML LLM key is absent."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(_source(), encoding="utf-8")
    review = ml_review({"workspace_path": str(workspace), "csv_path": "agents/agent1/tests/fixtures/leaky_project/data.csv", "problem_type": "classification", "profile": {}})
    result = ml_fix({"workspace_path": str(workspace), "ml_review": review["ml_review"], "problem_type": "classification", "enable_ml_fixes": True, "baseline": {"stdout": ""}})
    updated = (workspace / "train.py").read_text(encoding="utf-8")
    assert "random_state=42" in updated
    assert "stratify=y" in updated
    assert "class_weight='balanced'" in updated or 'class_weight="balanced"' in updated
    assert "XAI_API_KEY" not in str(result)
