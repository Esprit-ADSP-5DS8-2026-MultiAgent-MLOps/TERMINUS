"""Étape 6a : dépôts hétérogènes annotés À LA MAIN + mesure du taux de faux rejets.
À lancer à la racine du dépôt. Crée eval/real/ (6 dépôts + annotations.json) et eval/real_eval.py. Idempotent.

Pourquoi : eval/annotations.json est circulaire (expected_status est calculé par le moteur de politique, et les projets suivent
les conventions du générateur). Ici, l'annotation est un JUGEMENT HUMAIN indépendant du moteur : « ce dépôt est-il acceptable ? »."""
import json
from pathlib import Path

ROOT = Path(".")
REAL = ROOT / "eval" / "real"
REPOS = REAL / "repos"

REQS = "scikit-learn==1.8.0\npandas==3.0.2\nnumpy==2.4.4\njoblib==1.5.3\n"


def credit_csv(n=240):
    s, rows = 12345, ["income,age,debt_ratio,default"]
    for _ in range(n):
        vals = []
        for _k in range(4):
            s = (s * 1103515245 + 12345) % (2 ** 31)
            vals.append(s)
        age, income, debt, noise = 20 + vals[0] % 45, 1500 + vals[1] % 6000, (vals[2] % 100) / 100, (vals[3] >> 8) % 100 / 250
        risk = debt * 2 - income / 6000 + (0.3 if age < 28 else 0.0) + noise
        rows.append(f"{income},{age},{debt:.2f},{int(risk > 0.45)}")
    return "\n".join(rows) + "\n"


FEATURES = '''"""Séparation variables / cible."""


def split_xy(df, target):
    return df.drop(columns=[target]), df[target]
'''

TRAINING = '''"""Entraînement du modèle de défaut de paiement."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .features import split_xy


def fit(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = split_xy(df, "default")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    pred = model.predict(X_te)
    return {"accuracy": float(accuracy_score(y_te, pred)), "f1": float(f1_score(y_te, pred))}
'''

INFERENCE = '''"""Chargement et prédiction."""
import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    df = df[list(model.feature_names_in_)]
    return [int(v) for v in model.predict(df)]
'''

TRAINING_DEFECTS = '''"""Entraînement (version avec défauts réels : chemin absolu en dur + métriques écrites à la main)."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import split_xy

REPORT_DIR = "/home/alice/reports/credit"   # chemin absolu de la machine de l'auteur


def fit(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = split_xy(df, "default")
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    return {"accuracy": 0.97, "f1": 0.95}
'''

TRAINER_FLAT = '''"""Entraînement : module à la racine, fonction nommée fit()."""
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def fit(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["default"]), df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=7).fit(X_tr, y_tr)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    return {"roc_auc": float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))}
'''

AMBIG_A = '''"""Expérience A : forêt aléatoire."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["default"]), df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    m = RandomForestClassifier(n_estimators=30, random_state=1).fit(X_tr, y_tr)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(m, Path(output_dir) / "model.joblib")
    return {"accuracy": float(m.score(X_te, y_te))}
'''
AMBIG_B = AMBIG_A.replace("Expérience A : forêt aléatoire.", "Expérience B : régression logistique.").replace(
    "from sklearn.ensemble import RandomForestClassifier", "from sklearn.linear_model import LogisticRegression").replace(
    "RandomForestClassifier(n_estimators=30, random_state=1)", "LogisticRegression(max_iter=500, random_state=1)")

NOTEBOOK = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Défaut de paiement : exploration et modèle"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["import pandas as pd\n", "from sklearn.linear_model import LogisticRegression\n", "from sklearn.model_selection import train_test_split\n",
                    "df = pd.read_csv('data.csv')\n", "df.describe()"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["X, y = df.drop(columns=['default']), df['default']\n", "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)\n",
                    "clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)\n", "print(clf.score(X_te, y_te))"]},
    ],
}


def manifest(train, predict, data="data/credit.csv"):
    return (f"project:\n  name: credit-default\n  task: classification\ndata:\n  path: {data}\n  target: default\n"
            f"evaluation:\n  primary_metric: accuracy\n  min_threshold: 0.6\nentrypoints:\n  train: {train}\n  predict: {predict}\n")


def src_pkg(training=TRAINING):
    return {"src/credit_model/__init__.py": "", "src/credit_model/features.py": FEATURES,
            "src/credit_model/training.py": training, "src/credit_model/inference.py": INFERENCE,
            "requirements.txt": REQS, "data/credit.csv": credit_csv(), "README.md": "# Credit default\n\nLayout `src/`, pas de `main.py`.\n"}


REPO_FILES = {
    "src_layout_manifest": {**src_pkg(), "mlops.yaml": manifest("credit_model.training:fit", "credit_model.inference")},
    "src_layout_no_manifest": src_pkg(),
    "flat_custom_names": {"trainer.py": TRAINER_FLAT, "serving.py": INFERENCE, "requirements.txt": REQS, "data.csv": credit_csv(),
                          "mlops.yaml": manifest("trainer:fit", "serving", "data.csv").replace("accuracy", "roc_auc")},
    "notebook_only": {"analysis.ipynb": json.dumps(NOTEBOOK, indent=1), "requirements.txt": REQS, "data.csv": credit_csv()},
    "src_layout_defects": {**src_pkg(TRAINING_DEFECTS), "mlops.yaml": manifest("credit_model.training:fit", "credit_model.inference")},
    "ambiguous_trainers": {"src/exps/__init__.py": "", "src/exps/run_a.py": AMBIG_A, "src/exps/run_b.py": AMBIG_B,
                           "src/exps/inference.py": INFERENCE, "requirements.txt": REQS, "data/credit.csv": credit_csv()},
}

ANNOTATIONS = {
    "_comment": ("Jugement HUMAIN, indépendant du moteur de politique. human_verdict : ACCEPTABLE (un relecteur l'admettrait tel quel, "
                 "donc le rejeter = FAUX REJET) | NOT_ACCEPTABLE (défaut réel) | REVIEW (ambigu : une revue humaine est la bonne issue). "
                 "acceptable_statuses : statuts du validateur compatibles avec ce jugement. expected_fail_checks : contrôles qui DOIVENT échouer. "
                 "target_after : statut visé une fois une étape de la feuille de route livrée. "
                 "Pour ajouter un VRAI dépôt : le cloner dans eval/real/repos/<nom> et ajouter une entrée ici (même format)."),
    "src_layout_manifest": {
        "human_verdict": "ACCEPTABLE", "acceptable_statuses": ["ACCEPTED", "NEEDS_REVIEW"], "expected_fail_checks": [],
        "rationale": "Layout src/, fit() déclaré dans entrypoints, pas de main.py, métriques calculées sur un holdout. Rien à reprocher.",
        "source": "fixture écrite à la main"},
    "src_layout_no_manifest": {
        "human_verdict": "ACCEPTABLE", "acceptable_statuses": ["ACCEPTED", "NEEDS_REVIEW"], "expected_fail_checks": [],
        "known_gaps": {"STRUCT.MANIFEST": "étape 2 (spec déduite sans mlops.yaml)"},
        "target_after": {"étape 2 (spec déduite)": "ACCEPTED"},
        "rationale": "Même code, sans mlops.yaml. Un humain déduit data/credit.csv, la cible `default` et la métrique en 30 secondes. Sans spec, NEEDS_REVIEW est tolérable ; REPAIR_REQUIRED/REJECTED est un faux rejet.",
        "source": "fixture écrite à la main"},
    "flat_custom_names": {
        "human_verdict": "ACCEPTABLE", "acceptable_statuses": ["ACCEPTED", "NEEDS_REVIEW"], "expected_fail_checks": [],
        "rationale": "Modules à la racine nommés trainer.py / serving.py, fonction fit(), entrypoints déclarés `trainer:fit` et `serving`.",
        "source": "fixture écrite à la main"},
    "notebook_only": {
        "human_verdict": "NOT_ACCEPTABLE", "acceptable_statuses": ["REPAIR_REQUIRED", "NEEDS_REVIEW"],
        "expected_fail_checks": ["STRUCT.REQUIRED_FILES"], "known_gaps": {"STRUCT.MANIFEST": "étape 2"},
        "rationale": "Seulement un notebook : aucune fonction d'entraînement ni de prédiction appelable. Doit être converti en modules (réparation). Ne doit JAMAIS être accepté.",
        "source": "fixture écrite à la main"},
    "src_layout_defects": {
        "human_verdict": "NOT_ACCEPTABLE", "acceptable_statuses": ["REPAIR_REQUIRED", "REJECTED"],
        "expected_fail_checks": ["IO.PATHS", "ML.METRICS_GENUINE"],
        "rationale": "Layout src/ correct, mais chemin absolu /home/alice/reports/... en dur et métriques écrites à la main (0.97 / 0.95). Les défauts doivent être trouvés AUSSI hors d'un train.py racine.",
        "source": "fixture écrite à la main"},
    "ambiguous_trainers": {
        "human_verdict": "REVIEW", "acceptable_statuses": ["NEEDS_REVIEW"], "expected_fail_checks": [],
        "known_gaps": {"STRUCT.MANIFEST": "étape 2"},
        "rationale": "Deux fonctions d'entraînement équivalentes (run_a.main, run_b.main) et aucun manifeste : on ne peut pas deviner laquelle est LA référence. Revue humaine.",
        "source": "fixture écrite à la main"},
}

REAL_EVAL = r'''#!/usr/bin/env python
"""Faux rejets sur dépôts hétérogènes annotés À LA MAIN (eval/real/annotations.json). Aucune clé LLM requise par défaut.

    python eval/real_eval.py                       # validateur v2, partie déterministe (venv par projet : réseau requis)
    python eval/real_eval.py --env current         # interpréteur courant (rapide ; ENV.SANDBOX_ISOLATION restera NEEDS_REVIEW)
    python eval/real_eval.py --baseline            # ajoute la baseline v1 (validator/) pour comparer
    python eval/real_eval.py --llm                 # avec le LLM (GROQ_API_KEY)
    python eval/real_eval.py --only src_layout     # sous-ensemble

Mesures : FAUX REJET = dépôt jugé ACCEPTABLE par un humain mais classé REPAIR_REQUIRED/REJECTED.
          FAUX ACCEPTÉ = dépôt NOT_ACCEPTABLE classé ACCEPTED.  Au niveau des contrôles : FAIL inattendus et défauts manqués.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

REJECTS = {"REPAIR_REQUIRED", "REJECTED"}


def run_v2(path, a):
    from agentic_core.llm_client import ScriptedClient, build_client
    from agentic_core.schemas import CheckStatus
    from validator_agent import ValidatorConfig, run_validator
    if a.llm:
        client, cfg = build_client("validator", a.model, a.llm_cache, False), ValidatorConfig(env=a.env)
    else:
        client, cfg = ScriptedClient([]), ValidatorConfig(env=a.env, max_llm_steps=0)
    rep = run_validator(path, client, "sklearn-tabular", cfg)
    fails = sorted({c.check_id for c in rep.checks if c.status == CheckStatus.FAIL})
    open_ = sorted({c.check_id for c in rep.checks if c.status == CheckStatus.NOT_CHECKED})
    return rep.status.value, fails, open_


def run_v1(path, a):
    from validator import validate
    return validate(path, bandit=False, env=a.env).verdict.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="venv")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model"), ap.add_argument("--llm-cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    ann = {k: v for k, v in json.loads((HERE / "real" / "annotations.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    rows = []
    for name, c in sorted(ann.items()):
        if a.only and not any(o in name for o in a.only):
            continue
        path = HERE / "real" / "repos" / name
        if not path.is_dir():
            print(f"(absent : {path})", file=sys.stderr)
            continue
        status, fails, open_ = run_v2(path, a)
        row = {"repo": name, "human": c["human_verdict"], "acceptable": c["acceptable_statuses"], "status": status,
               "fails": fails, "not_checked": open_,
               "known_gap_fails": sorted(set(fails) & set(c.get("known_gaps", {}))),
               "unexpected_fails": sorted(set(fails) - set(c["expected_fail_checks"]) - set(c.get("known_gaps", {}))),
               "missed_defects": sorted(set(c["expected_fail_checks"]) - set(fails)),
               "status_ok": status in c["acceptable_statuses"],
               "false_reject": c["human_verdict"] == "ACCEPTABLE" and status in REJECTS,
               "false_accept": c["human_verdict"] != "ACCEPTABLE" and status == "ACCEPTED"}
        if a.baseline:
            row["v1"] = run_v1(path, a)
        rows.append(row)
        print(f"  {name:26} {c['human_verdict']:14} -> {status:16} {'OK' if row['status_ok'] else 'ÉCART'}", flush=True)

    ok_rows = [r for r in rows if r["human"] == "ACCEPTABLE"]
    print("\n| Dépôt | Jugement humain | Statut v2 |" + (" Baseline v1 |" if a.baseline else "") + " FAIL inattendus | Défauts manqués |")
    print("|---|---|---|" + ("---|" if a.baseline else "") + "---|---|")
    for r in rows:
        flag = " ✖ FAUX REJET" if r["false_reject"] else (" ✖ FAUX ACCEPTÉ" if r["false_accept"] else ("" if r["status_ok"] else " ≠"))
        print(f"| {r['repo']} | {r['human']} | {r['status']}{flag} |" + (f" {r['v1']} |" if a.baseline else "")
              + f" {', '.join(r['unexpected_fails']) or '-'} | {', '.join(r['missed_defects']) or '-'} |")
    fr = sum(r["false_reject"] for r in rows)
    print(f"\nFAUX REJETS : {fr}/{len(ok_rows)} dépôts acceptables"
          + (f"   |   baseline v1 : {sum(r['v1'] == 'REJECTED' for r in ok_rows)}/{len(ok_rows)}" if a.baseline else ""))
    print(f"FAUX ACCEPTÉS : {sum(r['false_accept'] for r in rows)}/{len([r for r in rows if r['human'] != 'ACCEPTABLE'])} dépôts non acceptables")
    print(f"Statut conforme au jugement humain : {sum(r['status_ok'] for r in rows)}/{len(rows)}")
    print(f"Défauts attendus manqués : {sum(len(r['missed_defects']) for r in rows)}   |   FAIL inattendus : {sum(len(r['unexpected_fails']) for r in rows)}"
          f"   |   lacunes connues encore présentes : {sum(len(r['known_gap_fails']) for r in rows)}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


n = 0
for repo, files in REPO_FILES.items():
    for rel, text in files.items():
        n += write(REPOS / repo / rel, text)
n += write(REAL / "annotations.json", json.dumps(ANNOTATIONS, indent=2, ensure_ascii=False) + "\n")
n += write(ROOT / "eval" / "real_eval.py", REAL_EVAL)

# la description du contrôle ne doit plus dire « main.py, train.py, predict.py » (étape 1 : fichiers déduits des entrypoints)
for cf in ("policy/contracts/sklearn_tabular.yaml",):
    p = ROOT / cf
    s = p.read_text(encoding="utf-8")
    old = 'description: "main.py, train.py, predict.py et requirements.txt présents à la racine du projet (preuve d\'absence : « absent:<fichier> »)."'
    new = ('description: "requirements.txt et les fichiers des entrypoints (train, load_model, predict) présents : lus dans `entrypoints` du manifeste, '
           'sinon trouvés par AST, sinon train.py/predict.py. main.py facultatif (preuve d\'absence : « absent:<fichier> »)."')
    if old in s:
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        n += 1
        print(f"+ {cf} (description)")
print(f"{n} fichier(s) écrit(s). Lance : python eval/real_eval.py --baseline")
