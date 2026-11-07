"""Le validateur est évalué sur 16 projets dont le verdict attendu est connu à l'avance."""
import json
import tempfile
from pathlib import Path

import pytest

from validator import ContractConfig, validate
from validator.check_structure import parse_requirements
from validator.sandbox import SubprocessSandbox

EVAL = Path(__file__).resolve().parents[1] / "eval"
EXPECTED = json.loads((EVAL / "expected_verdicts.json").read_text())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_verdict_and_codes(name):
    exp = EXPECTED[name]
    rep = validate(EVAL / exp["path"], bandit=False)
    codes = {f.code for f in rep.findings}
    assert rep.verdict.value == exp["verdict"], f"{name}: {rep.verdict} / {sorted(codes)}"
    for c in exp["codes"]:
        assert c in codes, f"{name}: code attendu {c} absent ({sorted(codes)})"


def test_reference_is_fully_clean_and_dynamic_ran():
    rep = validate(EVAL / "projects/credit_risk", bandit=False)
    assert rep.summary["BLOCKER"] == rep.summary["WARNING"] == rep.summary["CONFIRM"] == 0
    assert [l.status for l in rep.levels] == ["done"] * 5
    assert rep.observed_metrics["roc_auc"] > 0.75
    assert rep.facts["dynamic"]["deterministic"] is True


def test_regression_reference_is_clean():
    """Régression (rmse) : le seuil est une erreur maximale, un rmse plus bas que le seuil est bon."""
    rep = validate(EVAL / "projects/house_prices", bandit=False)
    assert rep.summary["BLOCKER"] == rep.summary["WARNING"] == 0, [f.code for f in rep.findings]
    assert rep.observed_metrics["rmse"] < 40


def test_waived_blocker_becomes_a_traced_warning():
    """Désaccord v1/v2 tranché par un humain : le bloquant reste visible, en avertissement, avec son nom."""
    cfg = ContractConfig(waived_codes=frozenset({"C6.UNPINNED", "C11.DANGEROUS_CALL"}), waived_by="Mohamed")
    rep = validate(EVAL / "projects/variants/02_unpinned_deps", cfg, bandit=False)
    unpinned = [f for f in rep.findings if f.code == "C6.UNPINNED"]
    assert unpinned and all(f.severity.value == "WARNING" and "levé par Mohamed" in f.message for f in unpinned)
    assert rep.verdict.value != "REJECTED"
    danger = validate(EVAL / "projects/variants/05_dangerous_call", cfg, bandit=False)
    assert danger.verdict.value == "REJECTED"          # jamais levé : le code serait exécuté


def test_code_is_not_executed_when_static_blockers_exist():
    rep = validate(EVAL / "projects/variants/05_dangerous_call", bandit=False)
    dyn = next(l for l in rep.levels if l.name == "dynamic")
    assert dyn.status == "skipped" and "pas exécuté" in dyn.reason


def test_dynamic_skipped_without_manifest():
    rep = validate(EVAL / "projects/variants/09_no_manifest", bandit=False)
    assert next(l for l in rep.levels if l.name == "dynamic").status == "skipped"


def test_contract_is_configurable_allow_xgboost():
    cfg = ContractConfig(allow_frameworks=frozenset({"xgboost"}))
    rep = validate(EVAL / "projects/variants/10_wrong_framework", cfg, dynamic=False, bandit=False)
    assert "C2.UNSUPPORTED_FRAMEWORK" not in {f.code for f in rep.findings}


def test_missing_directory_is_rejected_not_crashing():
    assert validate("/nonexistent/project").verdict.value == "REJECTED"


def test_report_is_valid_json_roundtrip():
    from validator.models import ValidationReport
    rep = validate(EVAL / "projects/variants/02_unpinned_deps", dynamic=False, bandit=False)
    assert ValidationReport.model_validate_json(rep.model_dump_json()).verdict == rep.verdict


def test_requirements_parser():
    rows = parse_requirements("# c\nnumpy==1.0\npandas>=2  # x\n-r other.txt\ngit+https://x/y.git\n")
    assert [(r[0], r[3]) for r in rows if r[3]] == [(4, "option"), (5, "option")]
    assert str(rows[0][2].specifier) == "==1.0"


# --- le sandbox lui-même ---------------------------------------------------------------------------
def test_sandbox_kills_infinite_loop():
    sb = SubprocessSandbox()
    with tempfile.TemporaryDirectory() as w:
        r = sb.run(["python", "-c", "while True: pass"], Path(w), Path(w), {}, timeout=2)
    assert r.timed_out and r.duration_s < 10


def test_sandbox_blocks_network_when_isolated():
    sb = SubprocessSandbox()
    if not sb.network_isolated:
        pytest.skip("unshare indisponible")
    code = "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)"
    with tempfile.TemporaryDirectory() as w:
        r = sb.run(["python", "-c", code], Path(w), Path(w), {}, timeout=10)
    assert r.returncode != 0 and not r.timed_out


def test_sandbox_environment_is_scrubbed():
    sb = SubprocessSandbox()
    code = "import os; print(sorted(k for k in os.environ if 'SECRET' in k or 'TOKEN' in k))"
    import os
    os.environ["MY_SECRET_TOKEN"] = "xyz"
    with tempfile.TemporaryDirectory() as w:
        r = sb.run(["python", "-c", code], Path(w), Path(w), {}, timeout=10)
    assert r.stdout.strip() == "[]"
