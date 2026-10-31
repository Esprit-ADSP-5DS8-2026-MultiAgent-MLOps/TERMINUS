"""Étapes du Validator et vérification de l'environnement isolé (ENV.SANDBOX_ISOLATION)."""
import sys

from agentic_core.env_check import inspect_environment, pinned_requirements
from agentic_core.llm_client import ScriptedClient
from agentic_core.sandbox_exec import derive_requirements
from agentic_core.schemas import AdmissionStatus as S, Check, CheckStatus as C
from helpers import CREDIT, FakeRunner
from policy import load_contract
from validator_agent.agent import ValidatorConfig, run_validator
from validator_agent.checks import merge_llm_over_python

HARNESS = {"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie",
           "contract_steps": {"import_train": "ok", "import_predict": "ok", "train": "ok", "load_model": "ok",
                              "predict_df": "ok", "predict_dict": "ok"},
           "deterministic": True, "files_written_outside_output_dir": []}
CFG = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0)


def _run(env=None, info=None):
    r = FakeRunner(harness=HARNESS)
    if env is not None:
        r.info = {**FakeRunner.info, "environment": {**FakeRunner.info["environment"], **env}}
    if info is not None:
        r.info = info
    return run_validator(CREDIT, ScriptedClient([]), "sklearn-tabular", CFG, runner=r)


def _chk(rep):
    return next(c for c in rep.checks if c.check_id == "ENV.SANDBOX_ISOLATION")


def test_report_lists_steps_in_order():
    rep = _run()
    names = [s["name"] for s in rep.steps]
    assert [s["n"] for s in rep.steps] == list(range(1, len(rep.steps) + 1))
    for expected in ("contrôles statiques", "environnement + exécution sandbox", "moteur de politique"):
        assert expected in names
    assert any(s["kind"] == "tool" and s["name"] == "run_in_sandbox" for s in rep.steps)
    assert names.index("contrôles statiques") < names.index("environnement + exécution sandbox") < names.index("moteur de politique")


def test_dedicated_venv_passes():
    rep = _run()
    assert _chk(rep).status == C.PASS and rep.status == S.ACCEPTED
    assert rep.environment["own_venv"] is True


def test_validator_own_interpreter_is_refused():
    rep = _run(env={"mode": "current", "own_venv": False, "shares_validator_env": True})
    assert _chk(rep).status == C.FAIL and rep.status == S.NEEDS_REVIEW
    assert any(i.check_id == "ENV.SANDBOX_ISOLATION" for i in rep.issues)


def test_pinned_version_mismatch_is_refused():
    rep = _run(env={"mismatches": [{"package": "scikit-learn", "required": "==1.8.0", "installed": "1.9.1"}]})
    assert _chk(rep).status == C.FAIL and rep.status == S.NEEDS_REVIEW


def test_no_environment_information_is_not_checked():
    rep = _run(info={"backend": "fake", "network_isolated": True})
    assert _chk(rep).status == C.NOT_CHECKED and rep.status == S.NEEDS_REVIEW


def test_llm_cannot_decide_isolation():
    py = [Check(check_id="ENV.SANDBOX_ISOLATION", status=C.NOT_CHECKED, source="python")]
    llm = [Check(check_id="ENV.SANDBOX_ISOLATION", status=C.PASS, source="llm")]
    assert merge_llm_over_python(py, llm)[0].status == C.NOT_CHECKED
    assert merge_llm_over_python([], llm) == []


def test_inspect_environment_detects_validator_interpreter(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("packaging==0.0.1\n")
    env = inspect_environment(mode="current", backend="subprocess", python=sys.executable, dep_file=req,
                              deps_source="requirements.txt", network_isolated=False)
    assert env["shares_validator_env"] is True and env["own_venv"] is False
    assert env["pinned_checked"] == 1 and env["mismatches"]
    assert pinned_requirements(req) == {"packaging": "==0.0.1"}


def test_dependencies_derived_from_pyproject_and_environment_yml(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "pyproject.toml").write_text('[project]\nname="x"\ndependencies=["scikit-learn==1.5.0","torch==2.4.0"]\n')
    (b / "environment.yml").write_text("dependencies:\n  - python=3.12\n  - numpy=1.26.4\n  - pip:\n    - torch==2.4.0\n")
    ta, tb_ = derive_requirements(a, tmp_path).read_text(), derive_requirements(b, tmp_path).read_text()
    assert "# source: pyproject.toml" in ta and "scikit-learn==1.5.0" in ta
    assert "# source: environment.yml" in tb_ and "numpy==1.26.4" in tb_ and "torch==2.4.0" in tb_
    assert "python" not in tb_.split("\n", 1)[1]
    assert derive_requirements(tmp_path / "nothing", tmp_path) is None


def test_deep_learning_profile():
    c = load_contract("deep-learning")
    assert {"torch", "tensorflow", "keras"} <= set(c.frameworks_in_scope) and "torch" not in c.frameworks_out_of_scope
    assert c.sandbox_memory_mb == 16384 and c.harness == "train_only"
    for name in ("sklearn-tabular", "training-only", "deep-learning"):
        assert "ENV.SANDBOX_ISOLATION" in load_contract(name).mandatory_ids
