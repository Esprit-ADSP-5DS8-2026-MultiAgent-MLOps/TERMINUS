"""Judge Agent : 5 critères à preuves vérifiables, lecture seule, invariants, échec fermé, jamais de décision d'admission ; + jeu annoté (garde de périmètre)."""
import json
import shutil
import sys
from pathlib import Path

import pytest
from helpers import ROOT, calls, tc

sys.path.insert(0, str(ROOT / "eval"))
import judge_eval  # noqa: E402

from agentic_core.llm_client import LLMError, ScriptedClient  # noqa: E402
from agentic_core.schemas import CRITERIA, FailureKind, RepairReport, load_report, seal, verify_seal  # noqa: E402
from repair_agent.judge import JudgeConfig, run_judge  # noqa: E402
from repair_agent.judge.tools import JudgeToolBox  # noqa: E402

CASES = judge_eval.load_cases()
D = judge_eval.CASES


def verdict_args(verdict="ACCEPT", next_action="REVALIDATE", evidence="scikit-learn==1.8.0", results=None, **kw):
    results = results or {}
    return {"verdict": verdict, "next_action": next_action, "reason": "r", **kw,
            "criteria": [{"name": n, "result": results.get(n, "PASS"), "rationale": "ok", "evidence": evidence} for n in CRITERIA]}


def judge(case, script, model="judge-model", **kw):
    d = D / case
    rr = judge_eval.synthetic_repair_report(d)
    return run_judge(d / "original", rr, ScriptedClient(script, model=model), workspace=d / "repaired", **kw), rr


EXAMINE = [("list_changes", {}), ("get_diff", {}), ("get_test_results", {})]


def test_good_repair_is_accepted_with_grounded_evidence():
    v, rr = judge("good_pin_requirements", calls(*EXAMINE, ("submit_verdict", verdict_args())))
    assert v.verdict == "ACCEPT" and v.next_action == "REVALIDATE" and v.run_status.value == "OK" and v.overridden_by_invariant is None
    assert [c.name for c in v.criteria] == list(CRITERIA) and all(c.grounded for c in v.criteria)
    assert v.independent_model is True and verify_seal(v)
    assert [x.report_type for x in v.lineage] == ["repair"] and v.repair_id == rr.repair_id


def test_verdict_contains_no_admission_decision():
    v, _ = judge("good_pin_requirements", calls(*EXAMINE, ("submit_verdict", verdict_args())))
    assert not hasattr(v, "status") and not hasattr(v, "can_proceed")


def test_fabricated_evidence_is_refused_then_failed_closed():
    bad = verdict_args(evidence="cette ligne n'existe dans aucun diff")
    v, _ = judge("good_pin_requirements", calls(*EXAMINE, ("submit_verdict", bad), ("submit_verdict", bad), ("submit_verdict", bad)))
    assert v.verdict != "ACCEPT" and v.next_action in ("RETRY", "ESCALATE") and v.overridden_by_invariant and "[forcé par invariant]" in v.reason


def test_must_examine_the_diff_before_judging():
    v, _ = judge("good_pin_requirements", calls(("get_test_results", {}), ("submit_verdict", verdict_args()), ("list_changes", {}), ("submit_verdict", verdict_args(evidence="requirements.txt"))))
    assert v.llm["submit_retries"] == 1 and v.verdict == "ACCEPT"


def test_all_five_criteria_required_and_coherence():
    partial = verdict_args()
    partial["criteria"] = partial["criteria"][:3]
    v, _ = judge("good_pin_requirements", calls(*EXAMINE, ("submit_verdict", partial), ("submit_verdict", verdict_args(verdict="REVISE", next_action="RETRY"))))
    assert v.verdict == "REVISE" and v.llm["submit_retries"] == 1
    incoherent = verdict_args(verdict="ACCEPT", results={"correctness": "FAIL"})
    v2, _ = judge("good_pin_requirements", calls(*EXAMINE, ("submit_verdict", incoherent), ("submit_verdict", verdict_args(verdict="REJECT", next_action="ACCEPT"))))
    assert v2.verdict != "ACCEPT"


@pytest.mark.parametrize("name,kind", [("bad_hardcoded_metrics", "hardcoded_metric"), ("bad_modified_test", "protected_path"), ("bad_changed_model", "scientific_change"),
                                       ("bad_changed_split", "scientific_change"), ("bad_secret_added", "secret_added"), ("bad_data_modified", "protected_path")])
def test_cheating_repairs_can_never_be_accepted_even_by_a_complacent_llm(name, kind):
    """Le LLM du Judge approuve tout, 3 fois : l'infrastructure (garde de périmètre RECALCULÉE) l'en empêche."""
    ok = verdict_args(evidence="a/")                                   # preuve valable : sous-chaîne du diff
    v, _ = judge(name, calls(*EXAMINE, ("submit_verdict", ok), ("submit_verdict", ok), ("submit_verdict", ok)))
    assert v.verdict == "REVISE" and v.next_action == "ESCALATE" and v.overridden_by_invariant and "périmètre" in v.overridden_by_invariant
    assert JudgeToolBox(D / name / "original", D / name / "repaired", judge_eval.synthetic_repair_report(D / name)).flags


def test_judge_recomputes_scope_and_does_not_trust_repair_report():
    """Le Repair déclare « aucun signalement » ; le Judge voit quand même le test modifié."""
    d = D / "bad_modified_test"
    rr = judge_eval.synthetic_repair_report(d)
    assert rr.scope_flags == [] and rr.requires_human_approval is False
    tb = JudgeToolBox(d / "original", d / "repaired", rr)
    out = json.loads(tb.call("list_changes", {}))
    assert any(f["kind"] == "protected_path" and "MODIFIÉ malgré" in f["detail"] for f in out["independent_scope_flags"])


def test_forged_repair_report_that_hides_a_change_is_detected(tmp_path):
    d = D / "bad_hardcoded_metrics"
    rr = judge_eval.synthetic_repair_report(d)
    rr = rr.model_copy(update={"modifications": [m for m in rr.modifications if m.file == "requirements.txt"]})    # cache train.py
    tb = JudgeToolBox(d / "original", d / "repaired", seal(rr))
    assert not tb.diff_matches_report and any("diff réel" in b for b in tb.blockers_for_accept())


def test_semantic_defects_need_the_llm_and_it_can_reject_them():
    """bad_incomplete : aucun signalement mécanique ; un Judge LLM lucide le refuse (complétude)."""
    d = D / "bad_incomplete"
    assert not JudgeToolBox(d / "original", d / "repaired", judge_eval.synthetic_repair_report(d)).flags
    v, _ = judge("bad_incomplete", calls(*EXAMINE, ("submit_verdict", verdict_args("REVISE", "RETRY", results={"completeness": "FAIL"}, remaining_concerns=["scikit-learn toujours non épinglé"]))))
    assert v.verdict == "REVISE" and v.next_action == "RETRY" and v.remaining_concerns


def test_failing_tests_or_changed_behavior_block_accept(tmp_path):
    d = D / "good_pin_requirements"
    rr = judge_eval.synthetic_repair_report(d)
    from agentic_core.schemas import MetricsComparison, TestResult
    bad = seal(rr.model_copy(update={"tests": [TestResult(name="unit_tests", status="FAIL")]}))
    assert any("test" in b for b in JudgeToolBox(d / "original", d / "repaired", bad).blockers_for_accept())
    moved = seal(rr.model_copy(update={"metrics_comparison": MetricsComparison(status="COMPARED", behavior_changed=True, tolerance=0.01)}))
    assert any("comportement" in b for b in JudgeToolBox(d / "original", d / "repaired", moved).blockers_for_accept())
    none_pass = seal(rr.model_copy(update={"tests": []}))
    assert any("aucune preuve" in b for b in JudgeToolBox(d / "original", d / "repaired", none_pass).blockers_for_accept())


def test_tools_are_read_only_and_confined():
    d = D / "good_pin_requirements"
    tb = JudgeToolBox(d / "original", d / "repaired", judge_eval.synthetic_repair_report(d))
    assert set(tb.tools) == {"list_files", "list_changes", "get_diff", "read_original", "read_repaired", "get_test_results", "get_metrics_comparison", "get_validator_issues"}
    for bad in ("../x", "/etc/passwd"):
        assert "error" in tb.call("read_original", {"path": bad})
    before = {p: p.read_bytes() for p in (d / "repaired").rglob("*") if p.is_file()}
    tb.call("read_repaired", {"path": "requirements.txt"})
    tb.call("get_diff", {})
    assert before == {p: p.read_bytes() for p in (d / "repaired").rglob("*") if p.is_file()}
    assert "untrusted" in tb.call("read_original", {"path": "train.py"})


@pytest.mark.parametrize("err,kind", [(LLMError("q", "quota"), FailureKind.QUOTA), (LLMError("t", "timeout"), FailureKind.TIMEOUT)])
def test_llm_errors_fail_closed_never_accept(err, kind):
    v, _ = judge("good_pin_requirements", [tc("list_changes", {}), err])
    assert v.run_status.value == "FAILED_CLOSED" and v.failure.kind == kind and v.verdict == "REVISE" and v.next_action == "ESCALATE" and verify_seal(v)


def test_invalid_inputs_fail_closed(tmp_path):
    d = D / "good_pin_requirements"
    rr = judge_eval.synthetic_repair_report(d)
    tampered = load_report(json.loads(rr.model_dump_json()))
    tampered.status = "FAILED"
    assert run_judge(d / "original", tampered, ScriptedClient([])).failure.kind == FailureKind.INPUT_INVALID
    other = tmp_path / "o"
    shutil.copytree(d / "original", other)
    (other / "train.py").write_text("# autre\n")
    assert "empreinte" in run_judge(other, rr, ScriptedClient([]), workspace=d / "repaired").failure.message
    changed = tmp_path / "r"
    shutil.copytree(d / "repaired", changed)
    (changed / "main.py").write_text("# modifié après réparation\n")
    assert "copie corrigée a changé" in run_judge(d / "original", rr, ScriptedClient([]), workspace=changed).failure.message


def test_independence_of_models_is_recorded_and_can_be_enforced():
    v, _ = judge("good_pin_requirements", calls(*EXAMINE, ("submit_verdict", verdict_args())), model="eval-repair")      # même modèle que le Repair
    assert v.independent_model is False and any("biais" in c for c in v.remaining_concerns)
    d = D / "good_pin_requirements"
    v2 = run_judge(d / "original", judge_eval.synthetic_repair_report(d), ScriptedClient([], model="eval-repair"), cfg=JudgeConfig(require_distinct_model=True), workspace=d / "repaired")
    assert v2.failure and v2.verdict != "ACCEPT"


# ----------------------------------------------------------------------------- jeu annoté : garde de périmètre sans LLM
def test_annotated_judge_set_scope_guard_catches_cheats_without_false_alarms():
    results = []
    for name, c in CASES.items():
        flags = judge_eval.scope_scan(D / name)
        if c["label"] == "good":
            assert flags == set(), f"fausse alerte sur {name}: {flags}"
        for expected in c["expected_scope_flags"]:
            assert expected in flags, f"{name}: {expected} non détecté ({flags})"
        results.append({"name": name, "label": c["label"], "verdict": "REVISE" if flags else "ACCEPT"})
    from metrics import score_judge
    s = score_judge(results)
    mechanical_bad = [n for n, c in CASES.items() if c["label"] == "bad" and not c["needs_llm"]]
    assert set(s["false_accepts"]) == {n for n, c in CASES.items() if c["needs_llm"]}      # seules les erreurs sémantiques échappent à la garde
    assert len(mechanical_bad) >= 6 and s["false_reject_rate"] == 0.0
