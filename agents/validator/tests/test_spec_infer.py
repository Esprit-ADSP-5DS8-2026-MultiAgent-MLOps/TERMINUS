"""Étape 2 : spec déduite sans manifeste, vérifiée par Python ; métrique lue dans le dict renvoyé par train()."""
import sys

import pytest
from test_entrypoints import CONTRACT, make_src_project, needs_ml, resolve

from agentic_core.llm_client import ScriptedClient
from agentic_core.sandbox_exec import SandboxRunner, pick_metric
from agentic_core.schemas import CheckStatus as C
from agentic_core.schemas import ProjectSpec
from helpers import FakeRunner
from validator_agent import ValidatorConfig, run_validator
from validator_agent.checks import run_static_checks
from validator_agent.spec_infer import deduce_spec, target_candidates
from validator_agent.tools import ValidatorToolBox


def tbox(root):
    tb = ValidatorToolBox(root, CONTRACT, runner=FakeRunner())
    tb.call("list_files", {})
    return tb


def deduce(root, spec=None):
    tb = tbox(root)
    return deduce_spec(tb, spec, tb.entrypoint_resolution())


def test_data_target_task_deduced_without_manifest(tmp_path):
    s = deduce(make_src_project(tmp_path, manifest=False))
    assert (s.data_path, s.target, s.task) == ("data.csv", "target", "classification")
    assert set(s.deduced) >= {"data_path", "target", "task"} and "colonne présente" in s.deduced["target"]
    assert s.primary_metric is None                      # jamais devinée statiquement : lue à l'exécution


def test_manifest_values_always_win(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    s = deduce(root, ProjectSpec(target="f1", data_path="data.csv", primary_metric="accuracy"))
    assert s.target == "f1" and "target" not in s.deduced and s.primary_metric == "accuracy"


def test_only_holes_are_filled_in_incomplete_manifest(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "mlops.yaml").write_text("entrypoints:\n  train: mlpkg.training:fit\n  predict: mlpkg.inference\n")
    c = {x.check_id: x for x in run_static_checks(ToolBoxFor(root), CONTRACT).checks}
    assert c["STRUCT.MANIFEST"].status == C.WARNING and "déduits" in c["STRUCT.MANIFEST"].note


def ToolBoxFor(root):
    return ValidatorToolBox(root, CONTRACT, runner=FakeRunner())


def test_two_data_files_without_reference_is_ambiguous(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "other.csv").write_text("a,b\n1,2\n")
    s = deduce(root)
    assert s is None or s.data_path is None


def test_data_file_referenced_in_code_is_chosen_among_several(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "other.csv").write_text("a,b\n1,2\n")
    (root / "src/mlpkg/training.py").write_text((root / "src/mlpkg/training.py").read_text() + '\nDEFAULT_DATA = "data.csv"\n')
    assert deduce(root).data_path == "data.csv"


def test_output_dirs_are_not_data(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "outputs").mkdir()
    (root / "outputs/preds.csv").write_text("a,b\n1,2\n")
    assert deduce(root).data_path == "data.csv"


def test_target_from_drop_when_single_column(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").write_text(
        (root / "src/mlpkg/training.py").read_text().replace('X, y = build(df, "target")', 'X = df.drop(columns=["target"])\n    y = df["f1"] * 0'))
    assert deduce(root).target == "target"          # une seule colonne retirée de X, présente dans le CSV


def test_target_absent_from_csv_is_not_deduced(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "data.csv").write_text("f1,f2,label_x\n1,2,0\n3,4,1\n5,6,0\n")
    s = deduce(root)
    assert s.target is None or s.target != "target"


def test_two_verified_drop_columns_are_ambiguous_not_guessed(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "src/mlpkg/training.py").write_text(
        (root / "src/mlpkg/training.py").read_text().replace('X, y = build(df, "target")', 'X = df.drop(columns=["f1", "target"])\n    y = df.iloc[:, -1]'))
    s = deduce(root)
    assert s.target is None and "ambigu" in s.deduced.get("target?", "")


def test_regression_task_from_float_target(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "data.csv").write_text("f1,f2,target\n" + "\n".join(f"{i},{i % 3},{i * 0.37:.2f}" for i in range(60)) + "\n")
    (root / "src/mlpkg/training.py").write_text(
        (root / "src/mlpkg/training.py").read_text().replace("LogisticRegression", "LinearRegression").replace("accuracy_score", "r2_score"))
    assert deduce(root).task == "regression"


def test_task_conflict_between_data_and_estimator_gives_none(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "data.csv").write_text("f1,f2,target\n" + "\n".join(f"{i},{i % 3},{i * 0.37:.2f}" for i in range(60)) + "\n")   # numérique continu
    s = deduce(root)                                                                                                    # mais LogisticRegression importé
    assert s.target == "target" and s.task is None


def test_target_candidates_contexts():
    import ast
    t = ast.parse('def f(df, target_col="y"):\n    y = df["z"].astype(int)\n    X = df.drop(columns=["a"])\nTARGET = "k"\n')
    c = target_candidates(t, {})
    assert ("y", 1) in c["param/variable nommée target|label"] and ("k", 4) in c["param/variable nommée target|label"]
    assert [s for s, _ in c["y = df[...]"]] == ["z"] and [s for s, _ in c["X = df.drop(...)"]] == ["a"]


def test_static_checks_no_manifest_warning_not_fail(tmp_path):
    c = {x.check_id: x for x in run_static_checks(ToolBoxFor(make_src_project(tmp_path, manifest=False)), CONTRACT).checks}
    assert c["STRUCT.MANIFEST"].status == C.WARNING and "DÉDUITE" in c["STRUCT.MANIFEST"].note
    assert c["DATA.USABLE"].status == C.PASS


def test_no_manifest_and_undeducible_spec_stays_fail(tmp_path):
    root = make_src_project(tmp_path, manifest=False)
    (root / "other.csv").write_text("a,b\n1,2\n")
    c = {x.check_id: x for x in run_static_checks(ToolBoxFor(root), CONTRACT).checks}
    assert c["STRUCT.MANIFEST"].status == C.FAIL and c["DATA.USABLE"].status == C.NOT_CHECKED


def test_pick_metric_rules():
    assert pick_metric({"accuracy": 0.9}) == ("accuracy", "seule métrique numérique renvoyée par train()", ["accuracy"])
    assert pick_metric({"mse": 1.0, "roc_auc": 0.8, "n": 3})[0] == "roc_auc"
    assert pick_metric({"zeta": 1.0, "alpha": 2.0})[0] == "zeta"
    assert pick_metric({"a": "x", "b": float("nan")})[0] is None and pick_metric(None)[0] is None


@needs_ml
def test_harness_discovers_metric_from_real_train_return(tmp_path):
    root = make_src_project(tmp_path / "p", manifest=False)
    runner = SandboxRunner(root, sandbox="subprocess", env=sys.executable, timeout=120)
    res = runner.run("harness", profile="sklearn_tabular", data_path="data.csv", target="target", primary_metric=None)
    assert res["executed_ok"] is True and res["primary_metric"] == "accuracy" and res["metric_discovery"]["chosen"] == "accuracy"


@needs_ml
def test_declared_metric_missing_from_returned_dict_still_fails(tmp_path):
    root = make_src_project(tmp_path / "p", manifest=False)
    runner = SandboxRunner(root, sandbox="subprocess", env=sys.executable, timeout=120)
    res = runner.run("harness", profile="sklearn_tabular", data_path="data.csv", target="target", primary_metric="roc_auc")
    assert res["executed_ok"] is False


@needs_ml
def test_end_to_end_no_manifest_exec_run_passes_with_deduced_spec(tmp_path):
    root = make_src_project(tmp_path / "p", manifest=False)
    rep = run_validator(root, ScriptedClient([]), "sklearn-tabular", ValidatorConfig(env=sys.executable, sandbox="subprocess", max_llm_steps=0))
    c = {x.check_id: x for x in rep.checks}
    assert c["EXEC.RUN"].status == C.PASS and c["DATA.USABLE"].status == C.PASS and c["STRUCT.MANIFEST"].status == C.WARNING
    assert rep.project_spec.primary_metric == "accuracy" and {"data_path", "target", "primary_metric"} <= set(rep.project_spec.deduced)
