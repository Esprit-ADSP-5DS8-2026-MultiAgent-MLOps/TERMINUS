"""Contrôles déterministes, fusion LLM, porte de sécurité, erreurs 413 et origine sandbox."""
from typing import get_args

from agentic_core.llm_client import LLMError, ScriptedClient
from agentic_core.schemas import AdmissionStatus as S, Category, Check, CheckStatus as C, Issue, Nature, Severity
from helpers import CREDIT, FakeRunner, PROJECTS
from mini_project import REQ_LINE, make_mini_project
from policy import load_contract
from policy.engine import evaluate
from validator_agent.agent import Invariants, ValidatorConfig, _make_checker, prefetch_facts, run_validator
from validator_agent.checks import (DeterministicOutcome, apply_harness, classify_exec_error, collapse_cascades, inspect_train_metrics,
                                    merge_llm_over_python, run_static_checks)
from validator_agent.tools import ValidatorToolBox

CONTRACT = load_contract("sklearn-tabular")
NO_LLM = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0)


def _tb(root, runner=None):
    return ValidatorToolBox(root, CONTRACT, runner=runner or FakeRunner())


def test_valid_category_accepted_unknown_rejected_with_explicit_list(tmp_path):
    tb = _tb(make_mini_project(tmp_path / "proj"))
    prefetch_facts(tb)
    check = _make_checker(tb, CONTRACT, Invariants(require_all_mandatory_reported=False, cross_check_facts=False))
    checks = [{"check_id": CONTRACT.checks[0].id, "status": "PASS"}]
    issue = {"check_id": CONTRACT.checks[0].id, "description": "d", "evidence": REQ_LINE}
    _, problems = check({"checks": checks, "issues": [{**issue, "category": "configuration"}]})
    assert problems and "configuration" in problems[0]
    assert all(v in problems[0] for v in get_args(Category))
    sub, problems = check({"checks": checks, "issues": [{**issue, "category": "structure"}]})
    assert sub is not None and not [p for p in problems if "category" in p]


def test_llm_cannot_override_python_fail():
    py = [Check(check_id="CODE.SYNTAX", status=C.FAIL, source="python", evidence="x")]
    llm = [Check(check_id="CODE.SYNTAX", status=C.PASS, source="llm")]
    out = merge_llm_over_python(py, llm)
    assert out[0].status == C.FAIL and out[0].source == "python"


def test_mandatory_not_checked_forbids_accepted():
    cs = [Check(check_id=i, status=C.PASS) for i in CONTRACT.mandatory_ids if i != "EXEC.RUN"]
    cs.append(Check(check_id="EXEC.RUN", status=C.NOT_CHECKED))
    pol, _ = evaluate(CONTRACT, cs, [], executed_ok=True)
    assert pol.status == S.NEEDS_REVIEW and not pol.can_proceed


def test_recommended_fail_does_not_block_admission():
    cs = [Check(check_id=i, status=C.PASS) for i in CONTRACT.mandatory_ids]
    cs.append(Check(check_id="ENV.DEPENDENCIES_DECLARED", status=C.FAIL, evidence="x"))
    pol, out = evaluate(CONTRACT, cs, [], executed_ok=True)
    assert pol.status == S.ACCEPTED and "ENV.DEPENDENCIES_DECLARED" in pol.warning_checks
    assert next(c for c in out if c.check_id == "ENV.DEPENDENCIES_DECLARED").effective_status == C.WARNING


def test_blocking_security_prevents_execution(tmp_path):
    p = tmp_path / "evil"
    p.mkdir()
    (p / "main.py").write_text("import os\nos.system('echo x')\n")
    (p / "train.py").write_text("def train(data_path, output_dir):\n    return {'roc_auc': 0.9}\n")
    (p / "predict.py").write_text("def load_model(m):\n    return m\ndef predict(m, d):\n    return []\n")
    (p / "requirements.txt").write_text("scikit-learn==1.5.2\n")
    runner = FakeRunner()
    tb = _tb(p, runner)
    out = apply_harness(tb, CONTRACT, run_static_checks(tb, CONTRACT), auto_execute=True)
    assert out.security_blocked and runner.runs == []
    exec_c = next(c for c in out.checks if c.check_id == "EXEC.RUN")
    assert exec_c.status == C.NOT_CHECKED
    sec = next(c for c in out.checks if c.check_id == "SEC.DANGEROUS_CALLS")
    assert sec.status == C.FAIL


def test_execution_required_forbids_accepted_without_successful_run():
    cs = [Check(check_id=i, status=C.PASS) for i in CONTRACT.mandatory_ids]
    pol, _ = evaluate(CONTRACT, cs, [], executed_ok=False)
    assert pol.status == S.NEEDS_REVIEW and not pol.can_proceed


def test_sandbox_error_is_not_a_candidate_fail():
    res = {"not_run": True, "error": "sandbox indisponible : RuntimeError"}
    assert classify_exec_error(res, env="venv", declared={"scikit-learn"}) == "sandbox"
    missing = {"executed_ok": False, "stderr_tail": "ModuleNotFoundError: No module named 'joblib'"}
    assert classify_exec_error(missing, env="current", declared={"joblib", "scikit-learn"}) == "validator"
    assert classify_exec_error(missing, env="venv", declared={"scikit-learn"}) == "candidate"
    hidden = {"executed_ok": False, "_summary": "exécution NON réussie",
              "contract_steps": {"import_predict": "FAILED: ModuleNotFoundError: No module named 'joblib'"}}
    assert classify_exec_error(hidden, env="current", declared={"joblib"}) == "validator"


def test_http_413_never_yields_accepted(tmp_path):
    p = make_mini_project(tmp_path / "proj")
    (p / "train.py").write_text("def train(data_path, output_dir):\n    return metrics_from_somewhere()\n")
    client = ScriptedClient([LLMError("HTTP 413 Limit 8000 Requested 8568", "too_large")])
    r = run_validator(p, client, CONTRACT, ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=2),
                      runner=FakeRunner())
    assert r.run_status.value == "FAILED_CLOSED" and r.failure.kind.value == "QUOTA"
    assert r.status != S.ACCEPTED and not r.can_proceed
    assert any(c.status == C.NOT_CHECKED for c in r.checks)


def test_partial_report_marks_unchecked_controls(tmp_path):
    p = make_mini_project(tmp_path / "proj")
    r = run_validator(p, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner())
    unchecked = [c.check_id for c in r.checks if c.status == C.NOT_CHECKED]
    assert "EXEC.RUN" in unchecked or any(c.check_id == "STRUCT.REQUIRED_FILES" and c.status == C.FAIL for c in r.checks)
    assert r.status != S.ACCEPTED
    assert not any(c.status == C.PASS and c.check_id in CONTRACT.mandatory_ids and c.status == C.NOT_CHECKED for c in r.checks)


def test_same_evidence_and_policy_same_decision(tmp_path):
    p = make_mini_project(tmp_path / "proj")
    a = run_validator(p, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner())
    b = run_validator(p, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner())
    assert a.status == b.status
    assert [(c.check_id, c.status, c.effective_status) for c in a.checks] == [
        (c.check_id, c.status, c.effective_status) for c in b.checks]


def test_final_report_matches_schema_and_contract(tmp_path):
    r = run_validator(CREDIT, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner(
        harness={"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie",
                 "contract_steps": {"import_train": "ok", "import_predict": "ok", "train": "ok",
                                    "load_model": "ok", "predict_df": "ok", "predict_dict": "ok"},
                 "deterministic": True, "files_written_outside_output_dir": []}))
    ids = {c.check_id for c in r.checks}
    assert {c.id for c in CONTRACT.checks} <= ids
    assert r.llm.get("llm_invoked") is False and r.llm.get("llm_calls") == 0
    assert all(c.category in get_args(Category) for c in r.issues)
    assert r.policy.contract_id == "sklearn-tabular"
    assert r.status == S.ACCEPTED


def test_fake_metrics_literal_return_fails_despite_sklearn_imports():
    a3 = PROJECTS / "adversarial" / "A3_fake_metrics"
    out = run_static_checks(_tb(a3), CONTRACT)
    m = next(c for c in out.checks if c.check_id == "ML.METRICS_GENUINE")
    assert m.status == C.FAIL
    r = run_validator(a3, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner(
        harness={"executed_ok": True, "metrics": {"roc_auc": 0.99}, "_summary": "exécution réussie",
                 "contract_steps": {"import_train": "ok", "import_predict": "ok", "train": "ok",
                                    "load_model": "ok", "predict_df": "ok", "predict_dict": "ok"},
                 "deterministic": True, "files_written_outside_output_dir": []}))
    assert r.status == S.REPAIR_REQUIRED and r.llm.get("llm_invoked") is False
    assert next(c for c in r.checks if c.check_id == "ML.METRICS_GENUINE").status == C.FAIL


def test_genuine_metrics_are_pass_and_skip_llm():
    from validator_agent.agent import ValidatorConfig, _llm_needed
    out = run_static_checks(_tb(CREDIT), CONTRACT)
    m = next(c for c in out.checks if c.check_id == "ML.METRICS_GENUINE")
    assert m.status == C.PASS
    assert _llm_needed(ValidatorConfig(max_llm_steps=3), out) is False
    status, _, _ = inspect_train_metrics("def train(data_path, output_dir):\n    return {'roc_auc': 0.99}\n")
    assert status == C.FAIL


def test_joblib_missing_under_env_current_is_not_a_project_fail(tmp_path):
    p = make_mini_project(tmp_path / "proj")
    (p / "predict.py").write_text("import joblib\ndef load_model(m):\n    return m\ndef predict(m, d):\n    return []\n")
    (p / "mlops.yaml").write_text("data:\n  path: data.csv\n  target: target\nevaluation:\n  primary_metric: roc_auc\n")
    (p / "requirements.txt").write_text("scikit-learn==1.5.2\njoblib==1.5.3\npandas==2.2.2\n")
    (p / "train.py").write_text(
        "from sklearn.metrics import roc_auc_score\n"
        "def train(data_path, output_dir):\n    return {'roc_auc': roc_auc_score([0,1],[0.1,0.9])}\n")
    runner = FakeRunner(harness={"executed_ok": False, "_summary": "exécution NON réussie",
                                 "contract_steps": {"import_predict": "FAILED: ModuleNotFoundError: No module named 'joblib'"}},
                        env="current")
    tb = _tb(p, runner)
    out = apply_harness(tb, CONTRACT, run_static_checks(tb, CONTRACT), auto_execute=True)
    exec_c = next(c for c in out.checks if c.check_id == "EXEC.RUN")
    assert exec_c.status == C.NOT_CHECKED and exec_c.origin == "validator"
    pred = next(c for c in out.checks if c.check_id == "ML.PREDICT_INTERFACE")
    assert pred.status != C.FAIL


def test_uncertain_import_mapping_is_not_fail(tmp_path):
    p = make_mini_project(tmp_path / "proj")
    (p / "train.py").write_text("import mysterypkg\n" + (p / "train.py").read_text())
    tb = _tb(p)
    out = run_static_checks(tb, CONTRACT)
    decl = next(c for c in out.checks if c.check_id == "ENV.DEPENDENCIES_DECLARED")
    assert decl.status == C.NOT_CHECKED


def test_llm_cannot_turn_not_checked_exec_into_pass():
    py = [Check(check_id="EXEC.RUN", status=C.NOT_CHECKED, source="python")]
    llm = [Check(check_id="EXEC.RUN", status=C.PASS, source="llm")]
    out = merge_llm_over_python(py, llm)
    assert out[0].status == C.NOT_CHECKED


def test_report_issues_use_allowed_enums():
    issue = Issue(issue_id="I1", check_id="ENV.DEPENDENCIES_PINNED", category="dependencies",
                  severity=Severity.MAJOR, description="x", nature=Nature.CONFIGURATION)
    assert issue.category == "dependencies"


def test_deterministic_prompt_is_smaller_than_prefetch():
    from policy import render_contract_for_prompt
    from validator_agent.agent import INTERPRET, PROMPT, _user_message_deterministic
    tb = _tb(CREDIT)
    out = run_static_checks(tb, CONTRACT)
    open_ids = {c.check_id for c in out.checks if c.status == C.NOT_CHECKED}
    targeted = INTERPRET.replace("{{CONTRACT}}", render_contract_for_prompt(CONTRACT, open_ids))
    legacy = PROMPT.replace("{{CONTRACT}}", render_contract_for_prompt(CONTRACT)) + prefetch_facts(_tb(CREDIT))
    msg = _user_message_deterministic(CREDIT, CONTRACT, out, tb)
    assert "python_checks" in msg and "FAITS DÉJÀ COLLECTÉS" not in msg
    assert "contrôles encore ouverts" in INTERPRET and "HYPOTHÈSES" in INTERPRET
    assert len(targeted) + len(msg) < len(legacy)


def test_deterministic_cache_reuses_same_decision(tmp_path):
    cache = tmp_path / "cache"
    harness = {"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie",
               "contract_steps": {"import_train": "ok", "import_predict": "ok", "train": "ok",
                                  "load_model": "ok", "predict_df": "ok", "predict_dict": "ok"},
               "deterministic": True, "files_written_outside_output_dir": []}
    cfg = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0, cache_dir=str(cache), env="current")
    runner = FakeRunner(harness=harness)
    a = run_validator(CREDIT, ScriptedClient([]), CONTRACT, cfg, runner=runner)
    assert a.llm.get("deterministic_cache") is False and runner.runs
    runner2 = FakeRunner(harness=harness)
    b = run_validator(CREDIT, ScriptedClient([]), CONTRACT, cfg, runner=runner2)
    assert b.llm.get("deterministic_cache") is True and runner2.runs == []
    assert a.status == b.status == S.ACCEPTED
    assert [(c.check_id, c.status) for c in a.checks] == [(c.check_id, c.status) for c in b.checks]


def test_token_budget_deterministic_vs_explore():
    """Mesure hors LLM réel : taille de prompt et appels. Le mode déterministe n'appelle le LLM que si ML.METRICS_GENUINE est ouvert."""
    from policy import render_contract_for_prompt
    from validator_agent.agent import INTERPRET, PROMPT, _llm_needed, _user_message_deterministic
    tb = _tb(CREDIT)
    out = run_static_checks(tb, CONTRACT)
    open_ids = {c.check_id for c in out.checks if c.status == C.NOT_CHECKED}
    det = INTERPRET.replace("{{CONTRACT}}", render_contract_for_prompt(CONTRACT, open_ids))
    legacy = PROMPT.replace("{{CONTRACT}}", render_contract_for_prompt(CONTRACT))
    det_chars, legacy_chars = len(det) + len(_user_message_deterministic(CREDIT, CONTRACT, out, tb)), len(legacy) + len(prefetch_facts(_tb(CREDIT)))
    metrics_open = any(c.check_id == "ML.METRICS_GENUINE" and c.status == C.NOT_CHECKED for c in out.checks)
    assert _llm_needed(NO_LLM, out) is False
    assert _llm_needed(ValidatorConfig(deterministic=True, max_llm_steps=3), out) is metrics_open
    assert det_chars < legacy_chars
    assert not metrics_open  # credit_risk calcule vraiment les métriques : 0 appel LLM en mode déterministe


def test_cli_no_llm_and_explore_flags():
    from validator_agent.__main__ import build_parser, config_from_args
    a = build_parser().parse_args(["proj", "--json", "out.json", "--env", "current", "--no-llm"])
    cfg = config_from_args(a)
    assert a.json == "out.json" and a.env == "current"
    assert cfg.max_llm_steps == 0 and cfg.deterministic is True
    explore = config_from_args(build_parser().parse_args(["proj", "--explore", "--max-llm-steps", "8"]))
    assert explore.deterministic is False and explore.max_llm_steps == 8


# ----------------------------------------------------------------------------- cascades : une cause racine = un problème
def _issue(cid):
    return Issue(issue_id="", check_id=cid, description="d", evidence="e")


def test_missing_predict_is_one_root_cause_not_three():
    out = DeterministicOutcome(
        checks=[Check(check_id="STRUCT.REQUIRED_FILES", status=C.FAIL, evidence="absent:predict.py"),
                Check(check_id="ML.PREDICT_INTERFACE", status=C.FAIL, file="predict.py", evidence='{"import_predict": "FAILED"}'),
                Check(check_id="EXEC.RUN", status=C.FAIL, evidence="exécution NON réussie")],
        issues=[_issue("STRUCT.REQUIRED_FILES"), _issue("ML.PREDICT_INTERFACE"), _issue("EXEC.RUN")])
    collapse_cascades(CONTRACT, out)
    by = {c.check_id: c for c in out.checks}
    assert by["STRUCT.REQUIRED_FILES"].status == C.FAIL
    assert by["ML.PREDICT_INTERFACE"].status == by["EXEC.RUN"].status == C.NOT_CHECKED
    assert by["EXEC.RUN"].caused_by == ["STRUCT.REQUIRED_FILES"]
    assert [i.check_id for i in out.issues] == ["STRUCT.REQUIRED_FILES"]


def test_independent_failures_are_not_masked():
    out = DeterministicOutcome(
        checks=[Check(check_id="STRUCT.REQUIRED_FILES", status=C.FAIL, evidence="absent:main.py"),
                Check(check_id="ML.TRAIN_INTERFACE", status=C.FAIL, file="train.py", evidence='{"name": "train", "params": ["x"]}')],
        issues=[_issue("STRUCT.REQUIRED_FILES"), _issue("ML.TRAIN_INTERFACE")])
    collapse_cascades(CONTRACT, out)
    assert all(c.status == C.FAIL for c in out.checks) and len(out.issues) == 2


def test_cascade_is_not_resolved_by_llm():
    py = [Check(check_id="STRUCT.REQUIRED_FILES", status=C.FAIL, source="python"),
          Check(check_id="ML.PREDICT_INTERFACE", status=C.NOT_CHECKED, source="python", caused_by=["STRUCT.REQUIRED_FILES"])]
    llm = [Check(check_id="ML.PREDICT_INTERFACE", status=C.PASS, source="llm")]
    merged = {c.check_id: c for c in merge_llm_over_python(py, llm)}
    assert merged["ML.PREDICT_INTERFACE"].status == C.NOT_CHECKED


def test_missing_predict_end_to_end_single_issue_and_io_paths_decided():
    from helpers import VARIANTS
    harness = {"executed_ok": False, "_summary": "exécution NON réussie : voir contract_steps / main",
               "files_written_outside_output_dir": [], "main": {"returncode": 1, "stderr_tail": "ModuleNotFoundError: No module named 'predict'"},
               "contract_steps": {"import_train": "ok", "train": "ok", "import_predict": "FAILED: ModuleNotFoundError: No module named 'predict'"}}
    r = run_validator(VARIANTS / "01_missing_predict", ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner(harness))
    by = {c.check_id: c for c in r.checks}
    assert r.status == S.REPAIR_REQUIRED
    assert [i.check_id for i in r.issues] == ["STRUCT.REQUIRED_FILES"]
    assert by["IO.PATHS"].status == C.PASS
    assert by["EXEC.RUN"].caused_by == ["STRUCT.REQUIRED_FILES"]


def test_infra_failures_are_not_blamed_on_the_candidate():
    assert classify_exec_error({"executed_ok": False, "stderr_tail": "MemoryError"}, env="venv", declared=set()) == "sandbox"
    assert classify_exec_error({"executed_ok": False, "harness_error": "boom", "contract_steps": {}}, env="venv", declared=set()) == "harness"
    assert classify_exec_error({"executed_ok": False, "harness_error": "x", "contract_steps": {"train": "FAILED: e"}}, env="venv", declared=set()) == "candidate"
