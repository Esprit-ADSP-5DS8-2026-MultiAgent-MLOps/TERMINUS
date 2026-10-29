"""Résolution des entrypoints : manifeste -> AST -> convention ; fichiers requis déduits ; exécution réelle d'un layout src/."""
import json
import sys
from importlib import metadata

import pytest

from agentic_core.sandbox_exec import SandboxRunner
from agentic_core.schemas import CheckStatus as C
from helpers import FakeRunner
from policy import load_contract
from validator.entrypoints import module_location, read_manifest_body, resolve_entrypoints
from validator.fs import walk_project
from validator_agent.checks import required_files_for, run_static_checks
from validator_agent.tools import ValidatorToolBox

CONTRACT = load_contract("sklearn-tabular")


def _v(name, default):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return default


SK, PD, JL = _v("scikit-learn", "1.5.2"), _v("pandas", "2.2.2"), _v("joblib", "1.4.2")
HAS_ML = all(_v(n, None) for n in ("scikit-learn", "pandas", "joblib"))
needs_ml = pytest.mark.skipif(not HAS_ML, reason="scikit-learn/pandas/joblib absents de l'interpréteur courant (exécution réelle ignorée)")

FEATURES = "def build(df, target):\n    return df.drop(columns=[target]), df[target]\n"
FIT = '''import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .features import build


def fit(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = build(df, "target")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = LogisticRegression(random_state=0, max_iter=500).fit(Xtr, ytr)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(m, Path(output_dir) / "model.joblib")
    return {"accuracy": float(accuracy_score(yte, m.predict(Xte)))}
'''
INFER = '''import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else data
    return [int(x) for x in model.predict(df)]
'''


def make_src_project(root, manifest=True, extra=None):
    (root / "src" / "mlpkg").mkdir(parents=True)
    (root / "src" / "mlpkg" / "__init__.py").write_text("")
    (root / "src" / "mlpkg" / "features.py").write_text(FEATURES)
    (root / "src" / "mlpkg" / "training.py").write_text(FIT)
    (root / "src" / "mlpkg" / "inference.py").write_text(INFER)
    (root / "requirements.txt").write_text(f"scikit-learn=={SK}\npandas=={PD}\njoblib=={JL}\n")
    rows = ["f1,f2,target"] + [f"{i % 7},{(i * 3) % 5},{int((i % 7) + ((i * 3) % 5) > 5)}" for i in range(120)]
    (root / "data.csv").write_text("\n".join(rows) + "\n")
    if manifest:
        (root / "mlops.yaml").write_text(
            "project: {name: demo, task: classification}\ndata: {path: data.csv, target: target}\n"
            "evaluation: {primary_metric: accuracy, min_threshold: 0.5}\n"
            "entrypoints:\n  train: mlpkg.training:fit\n  predict: mlpkg.inference\n")
    for name, text in (extra or {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _two_experiments(root):
    (root / "src/mlpkg/training.py").unlink()
    (root / "src/mlpkg/exp_a.py").write_text("def run_a(data_path, output_dir):\n    return {}\n")
    (root / "src/mlpkg/exp_b.py").write_text("def run_b(data_path, output_dir):\n    return {}\n")
    return root


def resolve(root, harness="sklearn_tabular"):
    files, _ = walk_project(root)
    return resolve_entrypoints(root, files, read_manifest_body(root), harness)


def test_module_location_src_layout_and_flat():
    fl = {"src/pkg/__init__.py", "src/pkg/train.py", "train.py", "app/x.py"}
    assert module_location("src/pkg/train.py", fl) == ("pkg.train", "src")
    assert module_location("train.py", fl) == ("train", ".")
    assert module_location("app/x.py", fl) == ("x", "app")
    assert module_location("my-app/t.py", set()) == ("t", "my-app")
    assert module_location("my-app/pkg-x/t.py", {"my-app/pkg-x/__init__.py"})[0] is None   # nom non importable


def test_manifest_entrypoints_with_dotted_module_and_function(tmp_path):
    r = resolve(make_src_project(tmp_path))
    t, p, lm = r.entries["train"], r.entries["predict"], r.entries["load_model"]
    assert (t.file, t.function, t.module, t.sys_path, t.source) == ("src/mlpkg/training.py", "fit", "mlpkg.training", "src", "manifest")
    assert (p.file, p.function, lm.function, lm.module) == ("src/mlpkg/inference.py", "predict", "load_model", "mlpkg.inference")
    assert r.required_files(("train", "load_model", "predict")) == ["src/mlpkg/training.py", "src/mlpkg/inference.py"]
    assert not r.problems


def test_manifest_file_path_with_function(tmp_path):
    make_src_project(tmp_path, manifest=False)
    (tmp_path / "mlops.yaml").write_text("entrypoints:\n  train: src/mlpkg/training.py:fit\n  predict: src/mlpkg/inference.py\n")
    assert resolve(tmp_path).entries["train"].module == "mlpkg.training"


def test_ast_finds_fit_and_predict_pair_without_manifest(tmp_path):
    r = resolve(make_src_project(tmp_path, manifest=False))
    assert r.entries["train"].function == "fit" and r.entries["train"].source == "ast"
    assert r.entries["predict"].file == "src/mlpkg/inference.py" and r.entries["predict"].source == "ast"
    assert not r.problems


def test_load_model_reexported_by_import_is_accepted(tmp_path):
    make_src_project(tmp_path, manifest=False)
    (tmp_path / "src/mlpkg/loader.py").write_text("import joblib\n\n\ndef load_model(p):\n    return joblib.load(p)\n")
    (tmp_path / "src/mlpkg/inference.py").write_text("from .loader import load_model\n\n\ndef predict(model, data):\n    return []\n")
    r = resolve(tmp_path)
    assert r.entries["predict"].file == "src/mlpkg/inference.py" and "load_model" in r.entries


def test_two_training_functions_are_ambiguous_not_guessed(tmp_path):
    make_src_project(tmp_path, manifest=False)
    (tmp_path / "src/mlpkg/exp_a.py").write_text("def run_a(data_path, output_dir):\n    return {}\n")
    (tmp_path / "src/mlpkg/exp_b.py").write_text("def run_b(data_path, output_dir):\n    return {}\n")
    (tmp_path / "src/mlpkg/training.py").unlink()
    r = resolve(tmp_path)
    assert "train" not in r.entries and r.problem("train").kind == "ambiguous" and len(r.problem("train").candidates) == 2


def test_tests_and_conftest_are_never_entrypoints(tmp_path):
    make_src_project(tmp_path, manifest=False)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").write_text("def train(data_path, output_dir):\n    return {}\n")
    assert resolve(tmp_path).entries["train"].file == "src/mlpkg/training.py"


def test_convention_fallback_keeps_old_rejection_reason(tmp_path):
    (tmp_path / "train.py").write_text("def train(x):\n    return {}\n")
    r = resolve(tmp_path)
    assert r.entries["train"].source == "convention" and r.problem("predict").file == "predict.py"


def test_manifest_path_traversal_is_invalid(tmp_path):
    make_src_project(tmp_path, manifest=False)
    (tmp_path / "mlops.yaml").write_text("entrypoints:\n  train: ../../etc/x.py:fit\n")
    r = resolve(tmp_path)
    assert r.problem("train").kind == "invalid" and "train" not in r.entries


def test_manifest_declares_missing_file_or_function(tmp_path):
    make_src_project(tmp_path, manifest=False)
    (tmp_path / "mlops.yaml").write_text("entrypoints:\n  train: src/mlpkg/nope.py:fit\n  predict: src/mlpkg/inference.py:inexistante\n")
    r = resolve(tmp_path)
    assert r.problem("train").kind == "missing" and r.problem("train").file == "src/mlpkg/nope.py"
    assert r.problem("predict").kind == "missing"


def _static(root):
    tb = ValidatorToolBox(root, CONTRACT, runner=FakeRunner())
    out = run_static_checks(tb, CONTRACT)
    return {c.check_id: c for c in out.checks}


def test_static_checks_accept_src_layout_with_fit(tmp_path):
    c = _static(make_src_project(tmp_path))
    assert c["STRUCT.REQUIRED_FILES"].status == C.PASS
    assert c["ML.TRAIN_INTERFACE"].status == C.PASS and c["ML.TRAIN_INTERFACE"].file == "src/mlpkg/training.py"
    assert c["ML.PREDICT_INTERFACE"].status == C.PASS
    assert c["ML.METRICS_GENUINE"].status == C.PASS            # lit fit(), pas un train.py inexistant
    assert c["ENV.DEPENDENCIES_DECLARED"].status != C.FAIL      # mlpkg/features ne sont pas des paquets tiers


def test_static_checks_without_manifest_use_ast(tmp_path):
    c = _static(make_src_project(tmp_path, manifest=False))
    assert c["STRUCT.REQUIRED_FILES"].status == C.PASS and c["ML.TRAIN_INTERFACE"].status == C.PASS


def test_ambiguous_entrypoints_give_not_checked_not_fail(tmp_path):
    _two_experiments(make_src_project(tmp_path, manifest=False))
    c = _static(tmp_path)
    assert c["ML.TRAIN_INTERFACE"].status == C.NOT_CHECKED and c["STRUCT.REQUIRED_FILES"].status == C.NOT_CHECKED


def test_missing_train_still_reports_absent_file(tmp_path):
    (tmp_path / "requirements.txt").write_text("scikit-learn==1.5.2\npandas==2.2.2\n")
    (tmp_path / "predict.py").write_text("def load_model(m):\n    return m\n\n\ndef predict(m, d):\n    return []\n")
    c = _static(tmp_path)
    assert c["STRUCT.REQUIRED_FILES"].status == C.FAIL and "absent:train.py" in c["STRUCT.REQUIRED_FILES"].evidence


def test_main_py_is_optional_unless_declared(tmp_path):
    root = make_src_project(tmp_path)
    assert "main.py" not in required_files_for(CONTRACT, resolve(root))[0]


@needs_ml
def test_real_execution_src_layout_with_fit_through_the_harness(tmp_path):
    root = make_src_project(tmp_path / "proj")
    runner = SandboxRunner(root, sandbox="subprocess", env=sys.executable, timeout=120)
    res = runner.run("harness", profile="sklearn_tabular", data_path="data.csv", target="target", primary_metric="accuracy")
    assert not res.get("not_run"), res
    assert res["contract_steps"]["import_train"] == "ok" and res["contract_steps"]["train"] == "ok", res
    assert res["contract_steps"]["predict_df"] == "ok" and res["contract_steps"]["predict_dict"] == "ok", res
    assert res["executed_ok"] is True and res["entrypoints"]["train"] == "src/mlpkg/training.py:fit"
    assert res["metrics"]["accuracy"] > 0.5 and res["deterministic"] is True


def test_runner_refuses_to_run_when_entrypoints_are_ambiguous(tmp_path):
    root = _two_experiments(make_src_project(tmp_path / "proj", manifest=False))
    runner = SandboxRunner(root, sandbox="subprocess", env=sys.executable, timeout=60)
    res = runner.run("harness", profile="sklearn_tabular", data_path="data.csv", target="target", primary_metric="accuracy")
    assert res["not_run"] and "ambigus" in res["error"]


@needs_ml
def test_legacy_flat_project_still_runs(tmp_path):
    (tmp_path / "train.py").write_text("import joblib, pandas as pd\nfrom pathlib import Path\nfrom sklearn.dummy import DummyClassifier\n"
                                       "def train(data_path, output_dir):\n    df = pd.read_csv(data_path)\n    m = DummyClassifier().fit(df[['f1']], df['target'])\n"
                                       "    Path(output_dir).mkdir(exist_ok=True)\n    joblib.dump(m, Path(output_dir) / 'model.joblib')\n    return {'accuracy': 0.5}\n")
    (tmp_path / "predict.py").write_text(INFER.replace("model.predict(df)", "model.predict(df[['f1']])"))
    (tmp_path / "data.csv").write_text("f1,f2,target\n" + "\n".join(f"{i},{i},{i % 2}" for i in range(30)) + "\n")
    runner = SandboxRunner(tmp_path, sandbox="subprocess", env=sys.executable, timeout=120)
    res = runner.run("harness", profile="sklearn_tabular", data_path="data.csv", target="target", primary_metric="accuracy")
    assert res["executed_ok"] is True and res["entrypoints"]["train"] == "train.py:train"
