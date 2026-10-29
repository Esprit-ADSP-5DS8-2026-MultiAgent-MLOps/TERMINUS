"""Point 3 : introspection (paquets installés, signatures) et hook d'audit à la place de listes figées."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_core.facts import static_facts
from agentic_core.introspection import (audit_findings, build_request, dist_for_import, import_status, seed_suspects,
                                        unknown_estimator_frameworks)
from agentic_core.llm_client import ScriptedClient
from agentic_core.schemas import AdmissionStatus as S
from agentic_core.schemas import CheckStatus as C
from helpers import CREDIT, VARIANTS, FakeRunner
from policy import load_contract
from validator_agent.agent import ValidatorConfig, run_validator
from validator_agent.tools import ValidatorToolBox

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract("sklearn-tabular")
NO_LLM = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0)

try:
    import sklearn  # noqa: F401
    HAS_SK = True
except ImportError:
    HAS_SK = False
needs_sk = pytest.mark.skipif(not HAS_SK, reason="scikit-learn absent de l'interpréteur courant")

# Une bibliothèque INVENTÉE : aucune liste du code ne la connaît, seule sa signature parle.
FANCY = {"modules": {"fancyml": ["fancyml"], "tqdm": ["tqdm"]},
         "symbols": {"fancyml.Booster": {"kind": "class", "has_fit": True, "params": ["depth", "seed"], "positional": ["depth", "seed"],
                                         "defaults": {"seed": "None"}},
                     "fancyml.Splitter": {"kind": "class", "has_fit": False, "params": ["shuffle", "random_state"],
                                          "positional": ["shuffle", "random_state"], "defaults": {"shuffle": "False", "random_state": "None"}},
                     "fancyml.Pinned": {"kind": "class", "has_fit": True, "params": ["seed"], "positional": ["seed"], "defaults": {"seed": "7"}},
                     "tqdm.tqdm": {"kind": "class", "has_fit": False, "params": ["iterable"], "positional": ["iterable"], "defaults": {}}}}


def call(symbol, kws=(), nargs=0, consts=None, starkw=False, line=1):
    return {"file": "train.py", "line": line, "symbol": symbol, "kws": sorted(kws), "starkw": starkw, "nargs": nargs, "star": False, "consts": consts or {}}


def test_seed_suspects_come_from_the_signature_not_from_a_name_list():
    assert [s["call"] for s in seed_suspects([call("fancyml.Booster")], FANCY, False)] == ["Booster"]
    assert seed_suspects([call("fancyml.Booster", kws=["seed"], consts={"seed": 3})], FANCY, False) == []
    assert seed_suspects([call("fancyml.Booster", kws=["seed"])], FANCY, False) == []                # variable : supposée fournie
    assert seed_suspects([call("fancyml.Booster", nargs=2)], FANCY, False) == []                     # graine passée en positionnel
    assert seed_suspects([call("fancyml.Booster", starkw=True)], FANCY, False) == []
    assert len(seed_suspects([call("fancyml.Booster", kws=["seed"], consts={"seed": None})], FANCY, False)) == 1   # seed=None explicite
    assert seed_suspects([call("fancyml.Booster")], FANCY, True) == []                                # graine globale posée
    assert seed_suspects([call("fancyml.Pinned")], FANCY, False) == []                                # la bibliothèque fixe son propre défaut


def test_shuffle_default_false_is_read_from_the_signature():
    assert seed_suspects([call("fancyml.Splitter")], FANCY, False) == []
    assert len(seed_suspects([call("fancyml.Splitter", kws=["shuffle"], consts={"shuffle": True})], FANCY, False)) == 1


def test_unknown_estimator_framework_needs_fit_and_is_not_in_the_contract_lists():
    calls = [call(s) for s in FANCY["symbols"]]
    assert unknown_estimator_frameworks(calls, FANCY, ["scikit-learn"], ["torch"]) == {"fancyml": ["fancyml.Booster", "fancyml.Pinned"]}
    assert unknown_estimator_frameworks(calls, FANCY, ["fancyml"], ["torch"]) == {}              # classé « dans le périmètre » par le contrat
    assert unknown_estimator_frameworks(calls, FANCY, ["scikit-learn"], ["fancyml"]) == {}        # classé « hors périmètre » : autre contrôle


def test_import_status_distinguishes_transitive_from_absent():
    rep = {"modules": {"sklearn": ["scikit-learn"], "scipy": ["scipy"], "ghost": []}}
    undeclared, unresolved = import_status({"sklearn": 1, "scipy": 1, "ghost": 1}, rep, {"scikit-learn"})
    assert undeclared == [{"module": "scipy", "dists": ["scipy"]}] and unresolved == ["ghost"]


def test_audit_findings_only_sanction_the_candidate():
    ev = [{"event": "socket.connect", "kind": "network", "origin": "candidate", "where": "a.py:3", "detail": "1.2.3.4:80"},
          {"event": "subprocess.Popen", "kind": "process", "origin": "library", "where": "backend/context.py:9", "detail": "lscpu"}]
    f = audit_findings({"events": ev})
    assert len(f["network"]) == 1 and not f["process"] and len(f["library"]) == 1


# ------------------------------------------------------------------------------------------------ faits statiques
def test_facts_collect_third_party_calls_but_the_llm_never_sees_them(tmp_path):
    (tmp_path / "train.py").write_text("import numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nnp.random.seed(1)\nm = RandomForestClassifier(n_estimators=10)\n")
    f = static_facts(tmp_path)
    assert "sklearn" in f["_third_party_modules"] and f["_global_seed"] is True and "missing_random_state" not in f
    assert {"sklearn.ensemble.RandomForestClassifier", "numpy.random.seed"} <= {c["symbol"] for c in f["_third_party_calls"]}
    assert build_request(f)["modules"] == ["numpy", "sklearn"]
    shown = json.loads(ValidatorToolBox(tmp_path, CONTRACT, runner=FakeRunner()).analyze_code())
    assert not [k for k in shown if k.startswith("_")]


@needs_sk
def test_import_to_distribution_comes_from_installed_metadata():
    assert dist_for_import("sklearn") == "scikit-learn" and dist_for_import("zz_not_installed") == "zz-not-installed"


# ------------------------------------------------------------------------------------------------ scripts exécutés dans le sandbox (réels)
@needs_sk
def test_real_introspection_reads_signatures_of_installed_packages(tmp_path):
    (tmp_path / "req.json").write_text(json.dumps({"modules": ["sklearn", "zz_ghost"], "symbols": [
        "sklearn.ensemble.RandomForestClassifier", "sklearn.model_selection.KFold", "zz_ghost.Foo"]}))
    subprocess.run([sys.executable, str(ROOT / "agentic_core" / "introspect_script.py"), "--request", str(tmp_path / "req.json"),
                    "--report", str(tmp_path / "rep.json")], check=True, cwd=tmp_path)
    r = json.loads((tmp_path / "rep.json").read_text())
    assert r["modules"] == {"sklearn": ["scikit-learn"], "zz_ghost": []} and "zz_ghost.Foo" in r["errors"]
    rf, kf = r["symbols"]["sklearn.ensemble.RandomForestClassifier"], r["symbols"]["sklearn.model_selection.KFold"]
    assert rf["has_fit"] and rf["defaults"]["random_state"] == "None" and kf["defaults"]["shuffle"] == "False"
    calls = [call("sklearn.ensemble.RandomForestClassifier"), call("sklearn.model_selection.KFold", line=2)]
    assert [s["call"] for s in seed_suspects(calls, r, False)] == ["RandomForestClassifier"]


def _run_audited(tmp_path, source, extra=None):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text(source)
    for rel, text in (extra or {}).items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    log = tmp_path / "audit.jsonl"
    subprocess.run([sys.executable, str(ROOT / "agentic_core" / "audit_hook.py"), str(log), str(proj), "main.py"], check=True, cwd=proj)
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def test_audit_hook_sees_network_and_process_hidden_from_the_ast(tmp_path):
    ev = _run_audited(tmp_path, """s = __import__("so" + "cket")
try:
    s.create_connection(("203.0.113.5", 80), timeout=0.2)
except Exception:
    pass
getattr(__import__("sub" + "process"), "run")(["echo", "hi"])
import socket
try:
    socket.socket().connect_ex(("127.0.0.1", 9))
except Exception:
    pass
""")
    cand = {(e["kind"], e["detail"]) for e in ev if e["origin"] == "candidate"}
    assert ("network", "203.0.113.5:80") in cand and ("process", "echo hi") in cand
    assert not [e for e in ev if "127.0.0.1" in e["detail"]]                       # boucle locale : ignorée


def test_audit_hook_attributes_library_calls_to_the_library(tmp_path):
    sp = tmp_path / "venv" / "lib" / "site-packages"
    ev = _run_audited(tmp_path, f"import sys\nsys.path.insert(0, {str(sp)!r})\nimport libx\nlibx.probe()\n",
                      extra={"venv/lib/site-packages/libx/__init__.py": "import subprocess\n\n\ndef probe():\n    subprocess.run(['echo', 'lib'])\n"})
    assert [e["origin"] for e in ev if e["kind"] == "process"] == ["library"]
    assert not audit_findings({"events": ev})["process"] and len(audit_findings({"events": ev})["library"]) == 1


# ------------------------------------------------------------------------------------------------ bout en bout (faux sandbox)
def _run(report=None, audit=None):
    harness = {"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie", **({"audit": audit} if audit else {})}
    others = {"introspect": {"command": "introspect", "introspected": True, **report}} if report else {}
    r = run_validator(CREDIT, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner(harness, others=others))
    return r, {c.check_id: c for c in r.checks}


def _credit_report(dist_of=lambda m: [{"sklearn": "scikit-learn"}.get(m, m)], has_fit=False):
    f = static_facts(CREDIT)
    return {"modules": {m: dist_of(m) for m in f["_third_party_modules"]},
            "symbols": {c["symbol"]: {"kind": "class", "has_fit": has_fit, "params": [], "positional": [], "defaults": {}} for c in f["_third_party_calls"]}}


def test_candidate_network_attempt_seen_only_at_runtime_rejects_the_project():
    r, c = _run(audit={"events": [{"event": "socket.connect", "kind": "network", "origin": "candidate", "where": "train.py:9", "detail": "203.0.113.5:80"}]})
    assert c["SEC.NETWORK"].status == C.FAIL and "203.0.113.5" in c["SEC.NETWORK"].evidence and r.status == S.REJECTED


def test_library_process_event_is_a_limit_not_a_failure():
    r, c = _run(audit={"events": [{"event": "subprocess.Popen", "kind": "process", "origin": "library", "where": "backend/context.py:9", "detail": "lscpu"}]})
    assert c["SEC.DANGEROUS_CALLS"].status == C.PASS and any("bibliothèques" in x for x in r.limits)


def test_declared_imports_are_confirmed_by_the_project_venv():
    _, c = _run(report=_credit_report())
    assert c["ENV.DEPENDENCIES_DECLARED"].status == C.PASS and "introspection du venv" in c["ENV.DEPENDENCIES_DECLARED"].note
    assert c["FRAMEWORK.SCOPE"].status == C.PASS


def test_import_absent_from_the_venv_is_an_undeclared_dependency():
    _, c = _run(report=_credit_report(dist_of=lambda m: [] if m == "joblib" else [{"sklearn": "scikit-learn"}.get(m, m)]))
    assert c["ENV.DEPENDENCIES_DECLARED"].status == C.FAIL and "joblib" in c["ENV.DEPENDENCIES_DECLARED"].evidence


def test_unknown_estimator_framework_needs_review_instead_of_silent_pass():
    r, c = _run(report=_credit_report(dist_of=lambda m: ["mysteryml"], has_fit=True))
    assert c["FRAMEWORK.SCOPE"].status == C.NOT_CHECKED and "mysteryml" in c["FRAMEWORK.SCOPE"].evidence and r.status != S.ACCEPTED


def test_missing_project_file_is_not_reported_as_an_undeclared_dependency():
    """Régression : main.py fait `from predict import ...` et predict.py manque -> cause racine = fichier absent, pas une dépendance non déclarée."""
    root = VARIANTS / "01_missing_predict"
    mods = static_facts(root)["_third_party_modules"]
    assert "predict" in mods                                                   # vu comme « tiers » faute de fichier local
    rep = {"modules": {m: ([] if m == "predict" else [{"sklearn": "scikit-learn"}.get(m, m)]) for m in mods}, "symbols": {}}
    harness = {"executed_ok": False, "_summary": "exécution NON réussie", "contract_steps": {}}
    r = run_validator(root, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner(harness, others={"introspect": {"command": "introspect", "introspected": True, **rep}}))
    c = {x.check_id: x for x in r.checks}
    assert c["STRUCT.REQUIRED_FILES"].status == C.FAIL and c["ENV.DEPENDENCIES_DECLARED"].status == C.PASS
    assert [i.check_id for i in r.issues] == ["STRUCT.REQUIRED_FILES"]
