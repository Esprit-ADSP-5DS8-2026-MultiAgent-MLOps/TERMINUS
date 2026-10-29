"""Moteur de politique : les règles viennent du contrat (données) ; changer de profil = changer de fichier."""
import pytest
import yaml

from agentic_core.schemas import AdmissionStatus as S, Check, CheckStatus as C, Issue, Severity
from policy import list_contracts, load_contract, load_repair_policy, render_contract_for_prompt
from policy.engine import evaluate
from policy.model import AdmissionContract


def checks(contract, **status):
    """Tous les obligatoires PASS, sauf ceux précisés."""
    out = []
    for i in contract.mandatory_ids:
        s = status.get(i.replace(".", "_"), C.PASS)
        out.append(Check(check_id=i, status=s, evidence="x" if s != C.PASS else None))
    return out


def run(contract, cs, issues=(), executed=True):
    return evaluate(contract, cs, list(issues), executed_ok=executed)[0]


def test_profiles_are_valid_data_files():
    assert {"sklearn-tabular", "training-only"} <= set(list_contracts())
    for name in list_contracts():
        c = load_contract(name)
        assert c.checks and c.mandatory_ids and c.harness
    assert "EXEC.RUN" in load_contract().mandatory_ids


def test_all_pass_is_accepted():
    c = load_contract()
    p = run(c, checks(c))
    assert p.status == S.ACCEPTED and p.can_proceed


def test_mandatory_fail_blocks_by_contract_on_fail():
    c = load_contract()
    assert run(c, checks(c, ENV_DEPENDENCIES_PINNED=C.FAIL)).status == S.REPAIR_REQUIRED
    assert run(c, checks(c, SEC_DANGEROUS_CALLS=C.FAIL)).status == S.REJECTED
    assert run(c, checks(c, FRAMEWORK_SCOPE=C.FAIL)).status == S.REJECTED
    assert run(c, checks(c, STRUCT_MANIFEST=C.FAIL)).status == S.NEEDS_REVIEW        # l'ancien NEEDS_CONFIRMATION
    both = run(c, checks(c, ENV_DEPENDENCIES_PINNED=C.FAIL, SEC_SECRETS=C.FAIL))
    assert both.status == S.REJECTED                                                 # priorité REJECTED > REPAIR_REQUIRED


def test_mandatory_not_checked_prevents_accepted():
    c = load_contract()
    p = run(c, checks(c, ML_PREDICT_INTERFACE=C.NOT_CHECKED))
    assert p.status == S.NEEDS_REVIEW and "ML.PREDICT_INTERFACE" in p.review_checks and not p.can_proceed


def test_missing_mandatory_check_counts_as_not_checked():
    c = load_contract()
    cs = [x for x in checks(c) if x.check_id != "SEC.NETWORK"]                       # le LLM a « sauté » un contrôle gênant
    p, out = evaluate(c, cs, [], executed_ok=True)
    assert p.status == S.NEEDS_REVIEW
    assert next(x for x in out if x.check_id == "SEC.NETWORK").status == C.NOT_CHECKED


def test_not_applicable_needs_a_valid_justification():
    base = load_contract()
    data = base.model_dump()
    for chk in data["checks"]:
        if chk["id"] == "DATA.USABLE":
            chk["not_applicable_when"] = [{"id": "NO_DATA_REQUIRED", "description": "le projet n'a pas de données locales"}]
    c = AdmissionContract.model_validate(data)
    ok = [Check(check_id=i, status=C.PASS) for i in c.mandatory_ids if i != "DATA.USABLE"]
    good = ok + [Check(check_id="DATA.USABLE", status=C.NOT_APPLICABLE, na_reason="NO_DATA_REQUIRED", evidence="absent:data.csv")]
    assert run(c, good).status == S.ACCEPTED
    for bad in (Check(check_id="DATA.USABLE", status=C.NOT_APPLICABLE, na_reason="BECAUSE_I_SAY_SO", evidence="x"),    # raison hors contrat
                Check(check_id="DATA.USABLE", status=C.NOT_APPLICABLE, na_reason="NO_DATA_REQUIRED"),                  # sans preuve
                Check(check_id="DATA.USABLE", status=C.NOT_APPLICABLE, evidence="x")):                                 # sans raison
        assert run(c, ok + [bad]).status == S.NEEDS_REVIEW
    assert run(load_contract(), [Check(check_id=i, status=C.NOT_APPLICABLE, na_reason="x", evidence="y") if i == "EXEC.RUN" else Check(check_id=i, status=C.PASS)
                                 for i in load_contract().mandatory_ids]).status == S.NEEDS_REVIEW   # aucune condition prévue = jamais valable


def test_recommended_fail_is_only_a_warning():
    c = load_contract()
    cs = checks(c) + [Check(check_id="ML.REPRODUCIBILITY", status=C.FAIL, evidence="x"), Check(check_id="SEC.COMMITTED_ARTIFACTS", status=C.NOT_CHECKED)]
    p, out = evaluate(c, cs, [], executed_ok=True)
    assert p.status == S.ACCEPTED and {"ML.REPRODUCIBILITY", "SEC.COMMITTED_ARTIFACTS"} <= set(p.warning_checks)   # (un recommandé non rapporté est aussi un avertissement)
    assert next(x for x in out if x.check_id == "ML.REPRODUCIBILITY").effective_status == C.WARNING


def test_critical_issue_blocks_even_without_failed_check():
    c = load_contract()
    sec = Issue(issue_id="I1", check_id="OTHER", category="security", severity=Severity.CRITICAL, description="x")
    assert run(c, checks(c), [sec]).status == S.REJECTED
    other = Issue(issue_id="I1", check_id="OTHER", category="code", severity=Severity.CRITICAL, description="x")
    assert run(c, checks(c), [other]).status == S.REPAIR_REQUIRED


def test_execution_required_by_contract():
    c = load_contract()
    assert run(c, checks(c), executed=False).status == S.NEEDS_REVIEW
    assert run(c, checks(c), executed=None).status == S.NEEDS_REVIEW


def test_changing_profile_changes_decision_without_touching_code():
    """Un projet sans predict.py/main.py : REPAIR_REQUIRED en sklearn-tabular, ACCEPTED en training-only — mêmes faits, autre fichier de contrat."""
    facts = {"STRUCT.REQUIRED_FILES": C.PASS, "ML.PREDICT_INTERFACE": C.FAIL}

    def report_for(contract):
        cs = [Check(check_id=i, status=facts.get(i, C.PASS), evidence="absent:predict.py" if i in facts and facts[i] == C.FAIL else None) for i in contract.mandatory_ids]
        cs.append(Check(check_id="ML.PREDICT_INTERFACE", status=C.FAIL, evidence="absent:predict.py"))     # ignoré par training-only (absent du catalogue)
        return run(contract, cs)
    assert report_for(load_contract("sklearn-tabular")).status == S.REPAIR_REQUIRED
    assert report_for(load_contract("training-only")).status == S.ACCEPTED


def test_adding_a_profile_is_adding_a_file(tmp_path):
    data = yaml.safe_load(open(load_contract.__globals__["DIR"] / "contracts" / "training_only.yaml"))
    data.update(id="my-profile", frameworks_in_scope=["torch"])
    data["checks"] = [k for k in data["checks"] if k["id"] != "FRAMEWORK.SCOPE"]
    f = tmp_path / "mine.yaml"
    f.write_text(yaml.safe_dump(data))
    c = load_contract(f)
    assert c.id == "my-profile" and "FRAMEWORK.SCOPE" not in c.mandatory_ids and "my-profile" in render_contract_for_prompt(c)
    only_exec = render_contract_for_prompt(c, {"EXEC.RUN"})
    assert "EXEC.RUN" in only_exec and "CODE.SYNTAX" not in only_exec


def test_invalid_contracts_are_rejected():
    d = load_contract().model_dump()
    d["checks"].append(d["checks"][0])
    with pytest.raises(ValueError):
        AdmissionContract.model_validate(d)
    d = load_contract().model_dump()
    d["harness"] = None
    with pytest.raises(ValueError):
        AdmissionContract.model_validate(d)


def test_repair_policy_is_data():
    p = load_repair_policy()
    assert set(p.auto_repair_natures) == {"mechanical", "configuration"}
    assert any("tests" in x for x in p.protected_paths) and any(x.endswith(".csv") for x in p.protected_paths)
