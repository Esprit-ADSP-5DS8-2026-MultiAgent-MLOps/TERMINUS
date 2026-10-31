"""Repair Agent : copie isolée, patchs/retour arrière, garde de périmètre, tests relancés par l'infrastructure, échecs = statuts."""
import json
import shutil

import pytest
from helpers import CREDIT, VARIANTS, FakeRunner, calls, tc, validator_report

from agentic_core.llm_client import LLMError, ScriptedClient
from agentic_core.schemas import AdmissionStatus as S, Issue, Nature, Severity, load_report, verify_seal
from repair_agent import RepairConfig, run_repair
from repair_agent.patches import PatchManager
from repair_agent.scope import ScopeGuard
from repair_agent.workspace import Workspace, WorkspaceError
from validator.handoff import compute_project_hash

PINNED = "pandas==3.0.2\nnumpy==2.4.4\nscikit-learn==1.8.0\njoblib==1.5.3\n"


def fixed(*ids, patches=("p0001",)):
    return ("submit_repair", {"results": [{"issue_id": i, "outcome": "fixed", "patch_ids": list(patches), "explanation": "ok"} for i in ids]})


def write_reqs(content=PINNED, i="I1"):
    return ("write_file", {"path": "requirements.txt", "content": content, "issue_id": i, "reason": "épingler"})


def repair(tmp_path, project, vr, script, **kw):
    out = tmp_path / "fixed"
    r = run_repair(project, vr, out, ScriptedClient(script), kw.pop("cfg", RepairConfig(run_exec_checks=False)), **kw)
    return r, out


# ----------------------------------------------------------------------------- scénario nominal
def test_repairs_in_isolated_copy_and_original_is_never_touched(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    h0 = compute_project_hash(src)[0]
    r, out = repair(tmp_path, src, validator_report(src), calls(("read_file", {"path": "requirements.txt"}), write_reqs(), ("run_checks", {}), fixed("I1")))
    assert r.status == "REPAIRED" and r.next_action == "JUDGE" and r.requires_judge and r.requires_revalidation and not r.requires_human_approval
    assert compute_project_hash(src)[0] == h0 == r.source_project_hash                      # original intact (empreinte avant/après)
    assert r.repaired_project_hash and r.repaired_project_hash != h0 and compute_project_hash(out)[0] == r.repaired_project_hash
    m = r.modifications[0]
    assert (m.file, m.issue_id, m.action, m.diff_ref, m.risk) == ("requirements.txt", "I1", "rewrite", "patches/p0001.diff", "low")
    assert "-scikit-learn>=1.0" in r.diff and (tmp_path / "fixed.repair" / "patches" / "p0001.diff").exists()
    assert {t.name: t.status for t in r.tests}["dependencies"] == "PASS" and verify_seal(r)
    assert [x.report_type for x in r.lineage] == ["validator"] and r.source_validator_status == "REPAIR_REQUIRED"


def test_tests_are_rerun_by_infrastructure_not_taken_from_the_llm(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    r, _ = repair(tmp_path, src, validator_report(src), calls(write_reqs("pandas>=1\n"), fixed("I1")))           # le LLM déclare « fixed » à tort
    assert r.status == "PARTIAL"                                                                              # mais le test dependencies échoue
    assert {t.name: t.status for t in r.tests}["dependencies"] == "FAIL"


def test_fixed_claim_without_patch_is_refused(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    r, _ = repair(tmp_path, src, validator_report(src), calls(fixed("I1"), fixed("I1"), fixed("I1")))
    assert r.status in ("ESCALATED", "FAILED") and not r.modifications


# ----------------------------------------------------------------------------- périmètre
def test_protected_paths_are_blocked_and_flagged_for_the_judge(tmp_path):
    src = CREDIT
    vr = validator_report(src)
    cheat = [("write_file", {"path": "tests/test_model.py", "content": "def test_ok(): assert True\n", "issue_id": "I1", "reason": "x"}),
             ("write_file", {"path": "data.csv", "content": "a,target\n1,1\n", "issue_id": "I1", "reason": "x"}),
             ("write_file", {"path": "train.py", "content": "def train(d, o):\n    return {'roc_auc': 0.99}\n", "issue_id": "I1", "reason": "x"})]
    r, out = repair(tmp_path, src, vr, calls(*cheat, write_reqs(), fixed("I1")))
    kinds = [(f.kind, f.file, f.blocked) for f in r.scope_flags]
    assert ("protected_path", "tests/test_model.py", True) in kinds and ("protected_path", "data.csv", True) in kinds
    assert any(k[0] == "hardcoded_metric" and k[2] for k in kinds)
    assert not (out / "tests").exists() and (out / "data.csv").read_text() == (src / "data.csv").read_text()      # rien n'a été écrit
    assert [m.file for m in r.modifications] == ["requirements.txt"]


def test_secret_added_is_blocked(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    r, out = repair(tmp_path, src, validator_report(src), calls(
        ("write_file", {"path": "config.py", "content": 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789"\n', "issue_id": "I1", "reason": "x"}), write_reqs(), fixed("I1")))
    assert any(f.kind == "secret_added" and f.blocked for f in r.scope_flags) and not (out / "config.py").exists()


def test_scientific_change_is_applied_but_flagged_and_requires_approval(tmp_path):
    src = CREDIT
    train = (src / "train.py").read_text()
    assert "test_size" in train, "le projet de référence a changé : adapter ce test"
    r, _ = repair(tmp_path, src, validator_report(src), calls(
        ("replace_in_file", {"path": "train.py", "old_str": "test_size=0.2", "new_str": "test_size=0.9", "issue_id": "I1", "reason": "x"}), fixed("I1")))
    assert any(f.kind == "scientific_change" and not f.blocked for f in r.scope_flags) and r.requires_human_approval
    assert r.modifications[0].risk == "scientific"


def test_adding_a_seed_is_a_mechanical_fix_not_a_scientific_change():
    g = ScopeGuard.default()
    b = "m = RandomForestClassifier(n_estimators=100)\nA, B = train_test_split(X, y, test_size=0.2)\n"
    a = "m = RandomForestClassifier(n_estimators=100, random_state=0)\nA, B = train_test_split(X, y, test_size=0.2, random_state=0)\n"
    assert g.scan_change("train.py", b, a) == []
    assert {f.detail.split("(")[1].split(")")[0] for f in g.scan_change("train.py", b, b.replace("RandomForestClassifier", "LogisticRegression"))} == {"model"}
    assert [f.kind for f in g.scan_change("train.py", b, b.replace("0.2", "0.5"))] == ["scientific_change"]
    assert g.check_write("tests/test_x.py").blocked and g.check_write("sub/test_y.py").blocked and g.check_write("conftest.py").blocked
    assert g.check_write("a/b/data.csv").blocked and g.check_write("train.py") is None


def test_write_is_confined_to_the_workspace(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    from repair_agent.tools import RepairToolBox
    ws = Workspace.create(src, tmp_path / "w")
    tb = RepairToolBox(ws, PatchManager(ws.root, ws.meta), ScopeGuard.default(), None)
    for bad in ("../escape.py", "/tmp/x.py", "a/../../x.py"):
        assert "error" in tb.call("write_file", {"path": bad, "content": "x", "issue_id": "I1", "reason": "x"})
    link = ws.root / "lnk"
    link.symlink_to("/tmp")
    assert "error" in tb.call("write_file", {"path": "lnk/x.py", "content": "x", "issue_id": "I1", "reason": "x"})
    assert ws.source_untouched()


# ----------------------------------------------------------------------------- patchs, retour arrière, copie
def test_patch_manager_history_and_rollback(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "a.txt").write_text("1\n")
    pm = PatchManager(tmp_path / "w", tmp_path / "m")
    p1 = pm.apply("a.txt", "2\n", issue_id="I1", reason="r")
    pm.apply("b.txt", "new\n", issue_id="I2", action="create")
    pm.apply("a.txt", "3\n", issue_id="I1")
    assert [m.patch_id for m in pm.modifications()] == ["p0001", "p0002", "p0003"] and "-1\n+2" in p1.diff.replace("\n\n", "\n")
    assert pm.rollback("p0002") == ["p0003", "p0002"]                    # annule aussi les suivants
    assert (tmp_path / "w" / "a.txt").read_text() == "2\n" and not (tmp_path / "w" / "b.txt").exists()
    pm.rollback_all()
    assert (tmp_path / "w" / "a.txt").read_text() == "1\n"
    assert [h.event for h in pm.history].count("rolled_back") == 3
    with pytest.raises(KeyError):
        pm.rollback("p0001")


def test_revert_patch_tool_removes_scientific_flag(tmp_path):
    r, out = repair(tmp_path, CREDIT, validator_report(CREDIT), calls(
        ("replace_in_file", {"path": "train.py", "old_str": "test_size=0.2", "new_str": "test_size=0.9", "issue_id": "I1", "reason": "x"}),
        ("revert_patch", {"patch_id": "p0001"}),
        write_reqs(PINNED, "I1"), fixed("I1", patches=("p0003",))))
    assert not any(f.kind == "scientific_change" for f in r.scope_flags) and "test_size=0.2" in (out / "train.py").read_text()


def test_workspace_rules(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    Workspace.create(src, tmp_path / "w1")
    with pytest.raises(WorkspaceError):
        Workspace.create(src, tmp_path / "w1")                           # non vide
    with pytest.raises(WorkspaceError):
        Workspace.create(src, src / "inside")                            # dans le projet
    with pytest.raises(WorkspaceError):
        Workspace.create(tmp_path / "nope", tmp_path / "w2")


def test_binary_artifacts_are_not_copied_and_count_as_fixed(tmp_path):
    p = tmp_path / "src"
    shutil.copytree(CREDIT, p)
    (p / "model.joblib").write_bytes(b"\x80binary")
    issues = [Issue(issue_id="I1", check_id="SEC.COMMITTED_ARTIFACTS", category="security", severity=Severity.MINOR, description="artefact", nature=Nature.MECHANICAL)]
    r, out = repair(tmp_path, p, validator_report(p, issues=issues), [])
    assert not (out / "model.joblib").exists() and r.removed_artifacts == ["model.joblib"] and r.status == "REPAIRED" and r.llm["tool_calls"] == 0


# ----------------------------------------------------------------------------- triage : qui répare quoi
def test_logic_methodological_and_security_issues_are_never_repaired_by_the_llm(tmp_path):
    p = VARIANTS / "02_unpinned_deps"
    issues = [Issue(issue_id="I1", check_id="ML.METRICS_GENUINE", category="ml_contract", description="métriques en dur", nature=Nature.METHODOLOGICAL),
              Issue(issue_id="I2", check_id="CODE.LOGIC", category="code", description="x", nature=Nature.LOGIC),
              Issue(issue_id="I3", check_id="SEC.SECRETS", category="security", severity=Severity.CRITICAL, description="x")]
    client = ScriptedClient([])
    r = run_repair(p, validator_report(p, issues=issues), tmp_path / "w", client, RepairConfig(run_exec_checks=False))
    assert r.status == "ESCALATED" and r.next_action == "ESCALATE" and r.requires_human_approval and client.calls == []     # le LLM n'a même pas été appelé
    assert {u.issue_id for u in r.unresolved_issues} == {"I1", "I2", "I3"} and all(u.needs_human for u in r.unresolved_issues)
    assert not (tmp_path / "w").exists()


def test_mixed_issues_repair_the_mechanical_ones_and_report_the_rest(tmp_path):
    p = VARIANTS / "02_unpinned_deps"
    issues = [Issue(issue_id="I1", check_id="ENV.DEPENDENCIES_PINNED", category="dependencies", description="x", nature=Nature.CONFIGURATION),
              Issue(issue_id="I2", check_id="ML.METRICS_GENUINE", category="ml_contract", description="x", nature=Nature.METHODOLOGICAL)]
    r, _ = repair(tmp_path, p, validator_report(p, issues=issues), calls(write_reqs(), fixed("I1")))
    assert r.status == "PARTIAL" and [u.issue_id for u in r.unresolved_issues] == ["I2"] and r.requires_human_approval


@pytest.mark.parametrize("status,expected", [(S.ACCEPTED, "NOTHING_TO_DO"), (S.REJECTED, "ESCALATED")])
def test_accepted_nothing_to_do_and_rejected_not_repaired(tmp_path, status, expected):
    p = VARIANTS / "02_unpinned_deps"
    r = run_repair(p, validator_report(p, status=status), tmp_path / "w", ScriptedClient([]), RepairConfig(run_exec_checks=False))
    assert r.status == expected and not r.modifications


# ----------------------------------------------------------------------------- erreurs = statuts
@pytest.mark.parametrize("err", [LLMError("limite", "quota"), LLMError("lent", "timeout")])
def test_llm_failure_rolls_back_everything(tmp_path, err):
    src = VARIANTS / "02_unpinned_deps"
    r, out = repair(tmp_path, src, validator_report(src), [tc("write_file", {"path": "requirements.txt", "content": PINNED, "issue_id": "I1", "reason": "x"}), err])
    assert r.status == "FAILED" and r.run_status.value == "FAILED_CLOSED" and r.failure.kind.value in ("QUOTA", "TIMEOUT")
    assert (out / "requirements.txt").read_text() == (src / "requirements.txt").read_text() and not r.requires_judge and r.next_action == "ESCALATE"
    assert r.repaired_project_hash is None and verify_seal(r)


def test_inputs_are_validated_as_statuses(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    vr = validator_report(src)
    tampered = load_report(json.loads(vr.model_dump_json()))
    tampered.status = S.REPAIR_REQUIRED
    tampered.project = "forgé"
    assert run_repair(src, tampered, tmp_path / "a", ScriptedClient([])).failure.kind.value == "INPUT_INVALID"
    other = tmp_path / "other"
    shutil.copytree(src, other)
    (other / "train.py").write_text("# changed\n")
    assert "empreinte" in run_repair(other, vr, tmp_path / "b", ScriptedClient([])).failure.message
    assert run_repair(src, "/nonexistent.json", tmp_path / "c", ScriptedClient([])).status == "FAILED"


def test_retry_with_judge_feedback_reaches_the_prompt(tmp_path):
    from agentic_core.schemas import JudgeVerdict, seal
    src = VARIANTS / "02_unpinned_deps"
    jd = seal(JudgeVerdict(verdict_id="j", repair_id="r", verdict="REVISE", next_action="RETRY", remaining_concerns=["numpy a disparu"], reason="incomplet"))
    seen = []

    class Spy(ScriptedClient):
        def chat(self, messages, tools=None):
            seen.append(messages[1]["content"])
            return super().chat(messages, tools)
    r = run_repair(src, validator_report(src), tmp_path / "w", Spy(calls(write_reqs(), fixed("I1"))), RepairConfig(run_exec_checks=False), attempt=2, judge_feedback=jd)
    assert "numpy a disparu" in seen[0] and r.attempt == 2 and r.status == "REPAIRED"


# ----------------------------------------------------------------------------- métriques avant/après
def test_metrics_comparison_flags_behavior_change(tmp_path):
    src = CREDIT
    vr = validator_report(src, project_spec={"task": "classification", "target": "target", "primary_metric": "roc_auc", "data_path": "data.csv"})

    class Seq(FakeRunner):
        def run(self, command, root=None, **p):
            if command == "harness":
                self.runs.append(root)
                return {"executed_ok": True, "metrics": {"roc_auc": 0.95 if len(self.runs) == 1 else 0.70}}   # après d'abord, puis avant
            return {"ok": True, "returncode": 0, "command": command, "skipped": "x"}
    cfg = RepairConfig()
    r, _ = repair(tmp_path, src, vr, calls(write_reqs(PINNED), fixed("I1")), cfg=cfg, runner=Seq())
    mc = r.metrics_comparison
    assert mc.status == "COMPARED" and mc.behavior_changed is True and mc.delta["roc_auc"] == pytest.approx(0.25) and r.requires_human_approval


def test_metrics_comparison_unavailable_without_spec(tmp_path):
    src = VARIANTS / "02_unpinned_deps"
    r, _ = repair(tmp_path, src, validator_report(src), calls(write_reqs(), fixed("I1")), cfg=RepairConfig(), runner=FakeRunner())
    assert r.metrics_comparison.status == "NOT_AVAILABLE"
