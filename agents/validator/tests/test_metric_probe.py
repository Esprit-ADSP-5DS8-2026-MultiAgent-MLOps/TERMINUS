"""Point 4 : ML.METRICS_GENUINE comportemental (étiquettes permutées, détection de fuite) au lieu de la seule lecture du code."""
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_core.behavior_probe import judge_metric
from agentic_core.llm_client import ScriptedClient
from agentic_core.sandbox_exec import SandboxRunner
from agentic_core.schemas import AdmissionStatus as S
from agentic_core.schemas import CheckStatus as C
from helpers import CREDIT, FakeRunner
from policy import load_contract
from validator_agent.agent import ValidatorConfig, run_validator

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract("sklearn-tabular")
NO_LLM = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0)

try:
    import joblib, pandas, sklearn  # noqa: F401,E401
    HAS_ML = True
except ImportError:
    HAS_ML = False
needs_ml = pytest.mark.skipif(not HAS_ML, reason="scikit-learn/pandas/joblib absents de l'interpréteur courant")

HEAD = "import json, pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import train_test_split\n"
HONEST = HEAD + """
def train(data_path, output_dir):
    df = pd.read_csv(data_path); X, y = df.drop(columns=["target"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = RandomForestClassifier(random_state=0).fit(Xtr, ytr)
    return {"roc_auc": float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))}
"""
LEAK_MEMORIZING = HEAD + """
def train(data_path, output_dir):
    df = pd.read_csv(data_path); X, y = df.drop(columns=["target"]), df["target"]
    m = RandomForestClassifier(random_state=0).fit(X, y)
    return {"roc_auc": float(roc_auc_score(y, m.predict_proba(X)[:, 1]))}
"""
LEAK_LOGREG = HEAD + """
def train(data_path, output_dir):
    df = pd.read_csv(data_path); X, y = df.drop(columns=["target"]), df["target"]
    m = LogisticRegression().fit(X, y)
    return {"roc_auc": float(roc_auc_score(y, m.predict_proba(X)[:, 1]))}
"""
CONSTANT = HEAD + """
def _fake():
    return 0.99
def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    RandomForestClassifier(random_state=0).fit(df.drop(columns=["target"]), df["target"])
    return {"roc_auc": _fake()}
"""
FROM_FILE = HEAD + """
def train(data_path, output_dir):
    return json.load(open("precomputed.json"))
"""
IGNORES_DATA_PATH = HONEST.replace("pd.read_csv(data_path)", 'pd.read_csv("data.csv")')


def _project(root: Path, train_src: str) -> Path:
    root.mkdir(parents=True)
    (root / "train.py").write_text(train_src)
    (root / "predict.py").write_text("import joblib\n\ndef load_model(p):\n    return joblib.load(p)\n\ndef predict(m, d):\n    return []\n")
    (root / "precomputed.json").write_text('{"roc_auc": 0.97}')
    random.seed(1)
    with open(root / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "c", "target"])
        for _ in range(300):
            a, b, c = random.random(), random.random(), random.random()
            w.writerow([round(a, 4), round(b, 4), round(c, 4), int(a + 0.5 * b + 0.3 * random.random() > 0.8)])
    return root


def _verdict_via_real_harness(tmp: Path, train_src: str) -> dict:
    """Le VRAI script du harnais (+ la sonde) lancé comme dans le sandbox : dossier harness/ contenant _harness.py et _probe.py."""
    proj = _project(tmp / "proj", train_src)
    (tmp / "harness").mkdir()
    shutil.copy(ROOT.parents[1] / "validator" / "harness.py", tmp / "harness" / "_harness.py")  # validator/ (v1) à la racine du dépôt
    shutil.copy(ROOT / "agentic_core" / "behavior_probe.py", tmp / "harness" / "_probe.py")
    (tmp / "m.json").write_text(json.dumps({"data_path": "data.csv", "target": "target", "task": "n/a", "primary_metric": "roc_auc"}))
    for o in ("o1", "o2", "o3"):
        (tmp / o).mkdir()
    subprocess.run([sys.executable, str(tmp / "harness" / "_harness.py"), "--project", str(proj), "--manifest", str(tmp / "m.json"),
                    "--out1", str(tmp / "o1"), "--out2", str(tmp / "o2"), "--out3", str(tmp / "o3"), "--report", str(tmp / "r.json")],
                   check=True, cwd=tmp, capture_output=True)
    r = json.loads((tmp / "r.json").read_text())
    return judge_metric("roc_auc", r.get("metrics"), r.get("perm_metrics"), r.get("probe"), r.get("probe_error"))


# ------------------------------------------------------------------------------------------------ jugement pur (mesures -> verdict)
PROBE_TRAIN = {"fit_label_hashes": ["210:aa"], "metric_calls": [{"fn": "roc_auc_score", "y": "210:aa", "value": 0.6}]}
PROBE_HOLDOUT = {"fit_label_hashes": ["210:aa"], "metric_calls": [{"fn": "roc_auc_score", "y": "90:bb", "value": 0.5}]}


def test_judge_label_independent_when_metric_is_unchanged_by_permutation():
    j = judge_metric("acc", {"acc": 0.97}, {"acc": 0.97})
    assert j["verdict"] == "label_independent" and "0.97" in j["evidence"]


def test_judge_leak_when_metric_call_uses_the_labels_given_to_fit():
    assert judge_metric("auc", {"auc": 0.98}, {"auc": 0.6}, PROBE_TRAIN)["verdict"] == "leak"


def test_judge_leak_wins_over_equality_for_models_that_memorize():
    probe = {"fit_label_hashes": ["210:aa"], "metric_calls": [{"fn": "roc_auc_score", "y": "210:aa", "value": 1.0}]}
    assert judge_metric("auc", {"auc": 1.0}, {"auc": 1.0}, probe)["verdict"] == "leak"


def test_judge_genuine_with_holdout_confirmation_and_without():
    g = judge_metric("auc", {"auc": 0.98}, {"auc": 0.5}, PROBE_HOLDOUT)
    assert g["verdict"] == "genuine" and g["holdout_confirmed"] is True
    g2 = judge_metric("auc", {"auc": 0.98}, {"auc": 0.5}, {"metric_calls": []})                 # métrique calculée à la main : réagit, non confirmée
    assert g2["verdict"] == "genuine" and g2["holdout_confirmed"] is False


def test_judge_weak_when_metric_barely_moves_and_inconclusive_when_nothing_to_compare():
    assert judge_metric("acc", {"acc": 0.995}, {"acc": 0.99})["verdict"] == "weak"                # classes très déséquilibrées
    assert judge_metric("acc", {"acc": 0.9}, None, None, "ValueError: x")["verdict"] == "inconclusive"
    assert judge_metric("acc", {"acc": 0.9}, {"other": 0.5})["verdict"] == "inconclusive"
    assert judge_metric(None, {"acc": 0.9}, {"acc": 0.5})["verdict"] == "inconclusive"


# ------------------------------------------------------------------------------------------------ exécutions réelles du harnais
@needs_ml
def test_real_harness_honest_holdout_metric_is_genuine(tmp_path):
    j = _verdict_via_real_harness(tmp_path, HONEST)
    assert j["verdict"] == "genuine" and j["holdout_confirmed"] is True


@needs_ml
def test_real_harness_detects_leak_even_when_the_model_memorizes_or_not(tmp_path):
    assert _verdict_via_real_harness(tmp_path / "a", LEAK_MEMORIZING)["verdict"] == "leak"
    assert _verdict_via_real_harness(tmp_path / "b", LEAK_LOGREG)["verdict"] == "leak"


@needs_ml
def test_real_harness_detects_constant_precomputed_and_ignored_data_path(tmp_path):
    for name, src in (("const", CONSTANT), ("file", FROM_FILE), ("ignore", IGNORES_DATA_PATH)):
        assert _verdict_via_real_harness(tmp_path / name, src)["verdict"] == "label_independent", name


@needs_ml
def test_sandbox_runner_exposes_the_verdict_in_the_harness_result(tmp_path):
    root = _project(tmp_path / "p", LEAK_LOGREG)
    res = SandboxRunner(root, sandbox="subprocess", env=sys.executable, timeout=180).run(
        "harness", profile="sklearn_tabular", data_path="data.csv", target="target", primary_metric="roc_auc")
    assert res["metric_probe"]["verdict"] == "leak" and res["perm_metrics"]["roc_auc"] < res["metrics"]["roc_auc"]


# ------------------------------------------------------------------------------------------------ intégration dans les contrôles (faux sandbox)
def _run(probe):
    harness = {"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie", "primary_metric": "roc_auc", "metric_probe": probe}
    r = run_validator(CREDIT, ScriptedClient([]), CONTRACT, NO_LLM, runner=FakeRunner(harness))
    return r, {c.check_id: c for c in r.checks}


def test_label_independent_metric_fails_the_project_even_if_static_reading_passed():
    r, c = _run({"verdict": "label_independent", "evidence": "roc_auc: réel=0.99 ; étiquettes permutées=0.99", "why": "la métrique « roc_auc » est identique"})
    assert c["ML.METRICS_GENUINE"].status == C.FAIL and c["ML.METRICS_GENUINE"].source == "harness" and r.status == S.REPAIR_REQUIRED
    assert any(i.check_id == "ML.METRICS_GENUINE" for i in r.issues)


def test_leak_fails_and_genuine_confirms_without_the_llm():
    r, c = _run({"verdict": "leak", "evidence": "roc_auc: réel=1 ; étiquettes permutées=1", "why": "calculée sur les étiquettes d'entraînement"})
    assert c["ML.METRICS_GENUINE"].status == C.FAIL
    r, c = _run({"verdict": "genuine", "evidence": "roc_auc: réel=0.98 ; étiquettes permutées=0.5", "why": "réagit aux étiquettes"})
    assert c["ML.METRICS_GENUINE"].status == C.PASS and c["ML.METRICS_GENUINE"].source == "harness"


def test_weak_metric_needs_review_and_inconclusive_leaves_the_static_result():
    r, c = _run({"verdict": "weak", "evidence": "acc: réel=0.995 ; étiquettes permutées=0.99", "why": "varie de moins de 2 %"})
    assert c["ML.METRICS_GENUINE"].status == C.NOT_CHECKED and r.status != S.ACCEPTED
    r, c = _run({"verdict": "inconclusive", "why": "entraînement sur étiquettes permutées impossible"})
    assert c["ML.METRICS_GENUINE"].source != "harness" and any("sonde de métrique" in x for x in r.limits)
