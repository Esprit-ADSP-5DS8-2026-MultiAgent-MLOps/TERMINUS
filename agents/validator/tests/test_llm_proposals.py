"""Point 5 : le LLM propose, Python vérifie. Le LLM ne peut faire passer aucun projet qui échoue."""
import json
import sys

import pytest
from test_entrypoints import CONTRACT, make_src_project, needs_ml

from agentic_core.llm_client import ScriptedClient
from agentic_core.schemas import AdmissionStatus as S
from agentic_core.schemas import Check
from agentic_core.schemas import CheckStatus as C
from helpers import FakeRunner, calls
from validator_agent import ValidatorConfig, run_validator
from validator_agent.agent import _llm_needed
from validator_agent.checks import apply_harness, merge_llm_over_python, run_static_checks
from validator_agent.proposals import Proposals, SpecProposal, open_tasks, verify_proposals
from validator_agent.tools import ValidatorToolBox

NO_LLM = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0)
WITH_LLM = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=3)
MAND = frozenset(CONTRACT.mandatory_ids)


def config_project(root):
    """Projet bon, mais la colonne cible est déclarée dans config.json : le code ne la cite pas, Python ne peut pas la déduire."""
    make_src_project(root, manifest=False)
    (root / "data.csv").write_text((root / "data.csv").read_text().replace("f1,f2,target", "f1,f2,default", 1))
    (root / "config.json").write_text(json.dumps({"target": "default", "test_size": 0.3}))
    tr = root / "src/mlpkg/training.py"
    tr.write_text(tr.read_text().replace('X, y = build(df, "target")', 'X, y = build(df, CONFIG["target"])')
                  .replace("from .features import build", "import json\n\nfrom .features import build\n\n"
                           "CONFIG = json.loads((Path(__file__).resolve().parents[2] / \"config.json\").read_text())"))
    return root


txt_project = config_project


def tbox(root, runner=None):
    tb = ValidatorToolBox(root, CONTRACT, runner=runner or FakeRunner())
    tb.call("list_files", {})
    return tb


def static(root, runner=None):
    tb = tbox(root, runner)
    return tb, run_static_checks(tb, CONTRACT)


def open_mandatory(root, runner=None):
    rep = run_validator(root, ScriptedClient([]), CONTRACT, NO_LLM, runner=runner or FakeRunner())
    return [c.check_id for c in rep.checks if c.status == C.NOT_CHECKED and c.check_id in MAND]


def submit(open_ids, proposals=None, status="NOT_CHECKED", extra_checks=(), **kw):
    # un PASS sur EXEC.* est déjà refusé par le vérificateur de soumission (invariant existant) : le LLM « menteur » du test l'évite
    checks = [{"check_id": i, "status": "NOT_CHECKED" if i.startswith("EXEC.") else status} for i in open_ids] + list(extra_checks)
    return calls(("submit_report", {"project_type": "tabular_ml", "checks": checks, **({"proposals": proposals} if proposals else {}), **kw}))


GOOD = {"spec": {"target": "default"}}


# ------------------------------------------------------------------ déclencheur
def test_clean_project_never_calls_the_llm(tmp_path):
    _, out = static(make_src_project(tmp_path))
    assert open_tasks(out) == [] and _llm_needed(WITH_LLM, out) is False


def test_undeducible_spec_triggers_llm_unless_disabled(tmp_path):
    _, out = static(txt_project(tmp_path))
    assert [t.kind for t in open_tasks(out)] == ["spec"] and _llm_needed(WITH_LLM, out) is True
    assert _llm_needed(NO_LLM, out) is False


def test_ambiguous_entrypoints_trigger_llm(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").unlink()
    (root / "src/mlpkg/exp_a.py").write_text("def run_a(data_path, output_dir):\n    return {}\n")
    (root / "src/mlpkg/exp_b.py").write_text("def run_b(data_path, output_dir):\n    return {}\n")
    _, out = static(root)
    assert "entrypoints" in [t.kind for t in open_tasks(out)]


def test_security_blocked_project_never_calls_the_llm(tmp_path):
    root = txt_project(tmp_path)
    (root / "src/mlpkg/features.py").write_text("import os\n\n\ndef build(df, target):\n    os.system('x')\n    return df, df[target]\n")
    _, out = static(root)
    out.security_blocked = True
    assert open_tasks(out) == []


# ------------------------------------------------------------------ fusion : le LLM ne produit jamais un PASS/FAIL obligatoire
def test_llm_pass_on_open_mandatory_check_is_ignored():
    py = [Check(check_id="DATA.USABLE", status=C.NOT_CHECKED, source="python")]
    llm = [Check(check_id="DATA.USABLE", status=C.PASS, source="llm", note="tout va bien")]
    out = merge_llm_over_python(py, llm, MAND)[0]
    assert out.status == C.NOT_CHECKED and "LLM (PASS)" in out.note


def test_llm_fail_or_warning_on_open_mandatory_check_does_not_decide():
    for st in (C.FAIL, C.WARNING):
        out = merge_llm_over_python([Check(check_id="DATA.USABLE", status=C.NOT_CHECKED, source="python")],
                                    [Check(check_id="DATA.USABLE", status=st, source="llm", evidence="x")], MAND)[0]
        assert out.status == C.NOT_CHECKED


def test_llm_only_mandatory_check_cannot_be_pass():
    out = merge_llm_over_python([], [Check(check_id="DATA.USABLE", status=C.PASS, source="llm")], MAND)[0]
    assert out.status == C.NOT_CHECKED


def test_llm_can_warn_or_open_review_on_python_pass_metrics_but_never_fail():
    py = [Check(check_id="ML.METRICS_GENUINE", status=C.PASS, source="python")]
    assert merge_llm_over_python(py, [Check(check_id="ML.METRICS_GENUINE", status=C.WARNING, source="llm", evidence="x")], MAND)[0].status == C.WARNING
    assert merge_llm_over_python(py, [Check(check_id="ML.METRICS_GENUINE", status=C.FAIL, source="llm", evidence="x")], MAND)[0].status == C.NOT_CHECKED


def test_llm_still_cannot_touch_python_fail():
    py = [Check(check_id="IO.PATHS", status=C.FAIL, source="python", evidence="x")]
    assert merge_llm_over_python(py, [Check(check_id="IO.PATHS", status=C.PASS, source="llm")], MAND)[0].status == C.FAIL


# ------------------------------------------------------------------ vérification des propositions
def verify(root, proposals):
    tb, out = static(root)
    return verify_proposals(tb, out, Proposals.model_validate(proposals))


def test_valid_spec_is_confirmed(tmp_path):
    v = verify(txt_project(tmp_path), GOOD)
    assert v.spec == {"target": "default"} and not v.rejected
    assert "citée dans config.json" in v.provenance["target"]


def test_missing_data_file_is_rejected(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "data.csv").unlink()                                   # plus aucun fichier de données : Python ne déduit rien
    v = verify(root, {"spec": {"data_path": "data/nope.csv"}})
    assert not v.spec and any("absent" in r for r in v.rejected)


def test_code_file_is_not_a_data_file(tmp_path):
    v = verify(txt_project(tmp_path), {"spec": {"data_path": "src/mlpkg/training.py"}})
    assert not v.spec and v.rejected


def test_target_column_absent_is_rejected(tmp_path):
    v = verify(txt_project(tmp_path), {"spec": {"target": "defaut_inexistant"}})
    assert not v.spec and any("absente" in r for r in v.rejected)


def test_target_present_but_never_referenced_by_code_is_rejected(tmp_path):
    v = verify(txt_project(tmp_path), {"spec": {"target": "f2"}})
    assert "target" not in v.spec and any("jamais référencée" in r for r in v.rejected)


def test_values_already_established_by_python_are_ignored(tmp_path):
    v = verify(make_src_project(tmp_path), {"spec": {"data_path": "autre.csv", "target": "f1"}})
    assert not v.spec and sum("déjà établi" in r for r in v.rejected) == 2


def test_wrong_task_is_rejected(tmp_path):
    v = verify(txt_project(tmp_path), {"spec": {"target": "default", "task": "regression"}})
    assert v.spec.get("target") == "default" and "task" not in v.spec and any("task" in r for r in v.rejected)


def test_ambiguous_data_is_a_suggestion_not_a_decision(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "other.csv").write_text("a,b\n1,2\n")
    v = verify(root, {"spec": {"data_path": "data.csv"}})
    assert not v.spec and any("non tranché" in s for s in v.suggestions)


def test_entrypoint_nonexistent_function_is_rejected_and_resolved_role_ignored(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").write_text("def entrainer(chemin, sortie):\n    return {}\n")
    v = verify(root, {"entrypoints": {"train": "src/mlpkg/training.py:inexistante", "predict": "src/mlpkg/inference.py:predict"}})
    assert not v.entrypoints and any("train" in r and "non confirmé" in r for r in v.rejected) and any("déjà résolu" in r for r in v.rejected)


def test_unresolved_entrypoint_with_valid_signature_is_confirmed(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").write_text("def entrainer(chemin, sortie):\n    return {}\n")
    v = verify(root, {"entrypoints": {"train": "src/mlpkg/training.py:entrainer"}})
    assert v.entrypoints["train"].function == "entrainer" and v.entrypoints["train"].source == "llm"


def test_train_signature_must_accept_two_arguments(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").write_text("def entrainer():\n    return {}\n")
    v = verify(root, {"entrypoints": {"train": "src/mlpkg/training.py:entrainer"}})
    assert not v.entrypoints and any("signature" in r for r in v.rejected)


def test_choice_between_equivalent_candidates_is_only_a_suggestion(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").unlink()
    (root / "src/mlpkg/exp_a.py").write_text("def run_a(data_path, output_dir):\n    return {}\n")
    (root / "src/mlpkg/exp_b.py").write_text("def run_b(data_path, output_dir):\n    return {}\n")
    v = verify(root, {"entrypoints": {"train": "src/mlpkg/exp_a.py:run_a"}})
    assert not v.entrypoints and any("non tranché" in s for s in v.suggestions)


# ------------------------------------------------------------------ l'agent : adoption, rejet, bornes
def run_llm(root, script, runner=None, cfg=WITH_LLM):
    client = ScriptedClient(script)
    return run_validator(root, client, CONTRACT, cfg, runner=runner or FakeRunner()), client


def test_valid_proposal_is_adopted_after_a_real_trial_and_traced(tmp_path):
    root = txt_project(tmp_path)
    ids = open_mandatory(root)
    rep, client = run_llm(root, submit(ids, GOOD))
    spec = rep.project_spec
    assert spec.data_path == "data.csv" and spec.target == "default"
    assert "par le LLM" in spec.deduced["target"]
    assert rep.llm["llm_invoked"] is True and rep.llm["proposals"]["accepted"]
    assert any("proposition du LLM acceptée" in x for x in rep.limits)
    exec_run = next(c for c in rep.checks if c.check_id == "EXEC.RUN")
    assert exec_run.status == C.PASS and exec_run.source in ("python", "harness")      # PASS posé par Python, jamais par le LLM


def test_proposal_is_dropped_when_the_trial_run_fails(tmp_path):
    root = txt_project(tmp_path)
    ids = open_mandatory(root)
    runner = FakeRunner(harness={"executed_ok": False, "_summary": "exécution NON réussie", "contract_steps": {"train": "FAILED: ValueError"}})
    rep, _ = run_llm(root, submit(ids, GOOD), runner=runner)
    assert rep.project_spec is None or not rep.project_spec.target      # la cible proposée n'a pas été retenue
    assert any("non concluant" in x for x in rep.llm["proposals"]["rejected"])
    assert rep.status == S.NEEDS_REVIEW and next(c for c in rep.checks if c.check_id == "EXEC.RUN").status == C.NOT_CHECKED


def test_llm_cannot_make_a_failing_project_pass(tmp_path):
    """Projet avec un vrai défaut (chemin absolu en dur) ET une spec non déductible : le LLM est appelé, affirme PASS partout,
    propose une spec valide, injecte une spec dans `project_spec`. Le défaut Python reste un FAIL, le verdict n'est pas ACCEPTED."""
    root = txt_project(tmp_path)
    (root / "src/mlpkg/training.py").write_text((root / "src/mlpkg/training.py").read_text() + '\nREPORT_DIR = "/home/alice/reports"\n')
    ids = open_mandatory(root)
    claims = [{"check_id": c.id, "status": "PASS"} for c in CONTRACT.checks if c.id not in ids]
    rep, client = run_llm(root, submit(ids, GOOD, status="PASS", extra_checks=claims,
                                       project_spec={"data_path": "evil.csv", "target": "x", "primary_metric": "accuracy"}))
    by = {c.check_id: c for c in rep.checks}
    assert rep.llm["llm_invoked"] is True and client.script == [] and rep.run_status.value == "OK"     # le LLM a VRAIMENT répondu, sans échec de boucle
    assert by["IO.PATHS"].status == C.FAIL and rep.status != S.ACCEPTED and rep.can_proceed is False
    assert rep.project_spec.data_path != "evil.csv"                  # le LLM n'écrit pas la spec du rapport
    assert all(by[i].status != C.PASS or by[i].source in ("python", "harness") for i in ids if i in by)


def test_llm_pass_claims_on_open_mandatory_checks_leave_needs_review(tmp_path):
    root = txt_project(tmp_path)
    ids = open_mandatory(root)
    rep, client = run_llm(root, submit(ids, None, status="PASS"))   # aucune proposition vérifiable, seulement des PASS affirmés
    assert rep.llm["llm_invoked"] is True and client.script == []
    assert rep.status == S.NEEDS_REVIEW
    assert all(next(c for c in rep.checks if c.check_id == i).status != C.PASS for i in ids)


def test_doubt_is_attached_to_the_open_check_as_a_note(tmp_path):
    root = txt_project(tmp_path)
    ids = open_mandatory(root)
    rep, _ = run_llm(root, submit(ids, {"doubts": [{"check_id": "EXEC.RUN", "explanation": "données dans un .txt non reconnu"}]}))
    c = next(c for c in rep.checks if c.check_id == "EXEC.RUN")
    assert c.status == C.NOT_CHECKED and "données dans un .txt" in c.note


def test_llm_has_no_execution_tool_during_the_loop(tmp_path):
    tb = tbox(txt_project(tmp_path))
    names = lambda: {s["function"]["name"] for s in tb.specs()}                     # noqa: E731
    assert "run_in_sandbox" in names()
    tb.llm_can_execute = False
    assert "run_in_sandbox" not in names() and "proposals" in json.dumps(tb.specs())
    out = json.loads(tb.call("run_in_sandbox", {"command": "harness", "data_path": "data.csv", "target": "default"}))
    assert "réservé à Python" in out["error"]


@needs_ml
def test_end_to_end_real_execution_confirms_llm_proposal(tmp_path):
    root = txt_project(tmp_path / "p")
    cfg = ValidatorConfig(deterministic=True, env=sys.executable, sandbox="subprocess", max_llm_steps=3)
    ids = [c.check_id for c in run_validator(root, ScriptedClient([]), CONTRACT, ValidatorConfig(env=sys.executable, sandbox="subprocess", max_llm_steps=0)).checks
           if c.status == C.NOT_CHECKED and c.check_id in MAND]
    rep = run_validator(root, ScriptedClient(submit(ids, GOOD)), CONTRACT, cfg)
    by = {c.check_id: c for c in rep.checks}
    assert by["EXEC.RUN"].status == C.PASS and by["DATA.USABLE"].status == C.PASS and by["STRUCT.MANIFEST"].status == C.WARNING
    assert rep.project_spec.primary_metric == "accuracy" and "par le LLM" in rep.project_spec.deduced["target"]


def test_llm_sourced_target_cannot_create_a_fail():
    from validator_agent.checks import DeterministicOutcome
    from validator_agent.proposals import demote_llm_dependent_fails
    from agentic_core.schemas import ProjectSpec
    out = DeterministicOutcome(spec=ProjectSpec(target="default", deduced={"target": "proposée par le LLM ; colonne présente"}),
                               checks=[Check(check_id="ML.METRICS_GENUINE", status=C.FAIL, source="harness"), Check(check_id="IO.PATHS", status=C.FAIL, source="python")])
    assert demote_llm_dependent_fails(out) == ["ML.METRICS_GENUINE"]
    by = {c.check_id: c.status for c in out.checks}
    assert by == {"ML.METRICS_GENUINE": C.NOT_CHECKED, "IO.PATHS": C.FAIL}            # seuls les contrôles dépendant de la cible sont rouverts
    out2 = DeterministicOutcome(spec=ProjectSpec(target="default", deduced={"target": "colonne conventionnelle"}),
                                checks=[Check(check_id="ML.METRICS_GENUINE", status=C.FAIL, source="harness")])
    assert demote_llm_dependent_fails(out2) == []                                      # une cible déduite par Python garde son FAIL
