"""Projets de classification tabulaire SAINS et variés (domaine, colonnes, modèles, hyperparamètres, graines).

Les projets sont générés avec la bibliothèque standard seulement (aucune donnée réelle) : ils servent de base à l'injection de défauts.
Même structure que `eval/projects/credit_risk` (train.py, predict.py, main.py, preprocessing.py, mlops.yaml, requirements.txt, data.csv).
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

REQUIREMENTS = "scikit-learn==1.8.0\npandas==3.0.2\nnumpy==2.4.4\njoblib==1.5.3\n"

DOMAINS = [
    {"name": "credit-risk", "target": "default", "num": ["duration", "amount", "age", "installment_rate", "num_credits", "income"],
     "cat": {"housing": ["own", "rent", "free"], "purpose": ["car", "education", "business", "furniture"], "job": ["skilled", "unskilled", "management"]}},
    {"name": "customer-churn", "target": "churned", "num": ["tenure", "monthly_fee", "support_calls", "data_usage", "age", "late_payments"],
     "cat": {"plan": ["basic", "plus", "premium"], "contract": ["monthly", "yearly", "two_year"], "region": ["north", "south", "east", "west"]}},
    {"name": "loan-approval", "target": "approved", "num": ["loan_amount", "term_months", "annual_income", "debt_ratio", "credit_score", "dependents"],
     "cat": {"employment": ["salaried", "self_employed", "retired"], "education": ["school", "bachelor", "master"], "area": ["urban", "suburban", "rural"]}},
    {"name": "insurance-claims", "target": "fraud", "num": ["claim_amount", "policy_age", "driver_age", "past_claims", "vehicle_age", "distance_km"],
     "cat": {"vehicle_type": ["sedan", "suv", "van", "truck"], "coverage": ["basic", "full"], "channel": ["agent", "online", "broker"]}},
    {"name": "patient-readmission", "target": "readmitted", "num": ["stay_days", "num_procedures", "age", "num_medications", "prior_visits", "lab_score"],
     "cat": {"admission": ["emergency", "referral", "elective"], "insurance": ["public", "private", "none"], "ward": ["a", "b", "c"]}},
    {"name": "employee-attrition", "target": "left_company", "num": ["years_at_company", "salary", "overtime_hours", "satisfaction", "age", "promotions"],
     "cat": {"department": ["sales", "tech", "hr", "ops"], "travel": ["rare", "frequent", "none"], "remote": ["yes", "no", "hybrid"]}},
]

MODEL_POOL = [
    ("logistic_regression", "LogisticRegression", "from sklearn.linear_model import LogisticRegression",
     lambda r, rs: f'LogisticRegression(max_iter={r.choice([1000, 2000, 3000])}, class_weight="balanced", random_state={rs})'),
    ("random_forest", "RandomForestClassifier", "from sklearn.ensemble import RandomForestClassifier",
     lambda r, rs: f'RandomForestClassifier(n_estimators={r.choice([100, 200, 300])}, max_depth={r.choice([4, 6, 8])}, class_weight="balanced", random_state={rs})'),
    ("gradient_boosting", "GradientBoostingClassifier", "from sklearn.ensemble import GradientBoostingClassifier",
     lambda r, rs: f'GradientBoostingClassifier(n_estimators={r.choice([100, 150])}, learning_rate={r.choice([0.05, 0.1])}, random_state={rs})'),
    ("extra_trees", "ExtraTreesClassifier", "from sklearn.ensemble import ExtraTreesClassifier",
     lambda r, rs: f'ExtraTreesClassifier(n_estimators={r.choice([100, 200])}, max_depth={r.choice([6, 8])}, random_state={rs})'),
]


def _write_data(path: Path, dom: dict, rng: random.Random, n: int) -> None:
    nums, cats, target = dom["num"], dom["cat"], dom["target"]
    w = {c: rng.choice([-1, 1]) * rng.uniform(0.6, 1.4) for c in nums}
    cat_eff = {c: {v: rng.uniform(-0.8, 0.8) for v in vals} for c, vals in cats.items()}
    scale = {c: (rng.uniform(5, 50), rng.uniform(2, 20)) for c in nums}          # (moyenne, écart-type) arbitraires
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([*nums, *cats, target])
        for _ in range(n):
            z, row = 0.0, []
            for c in nums:
                mu, sd = scale[c]
                x = rng.gauss(0, 1)
                z += w[c] * x
                row.append(round(max(mu + sd * x, 0.1), 2))
            for c, vals in cats.items():
                v = rng.choice(vals)
                z += cat_eff[c][v]
                row.append(v)
            p = 1 / (1 + math.exp(-1.6 * z))
            wr.writerow([*row, int(rng.random() < p)])


def make_base(dest: Path, seed: int) -> dict:
    """Crée un projet sain dans `dest` (dossier créé) et retourne ses métadonnées (domaine, cible, modèles, graine...)."""
    rng = random.Random(seed)
    dom = rng.choice(DOMAINS)
    nums = rng.sample(dom["num"], rng.randint(4, len(dom["num"])))
    cats = {k: dom["cat"][k] for k in rng.sample(list(dom["cat"]), rng.randint(1, len(dom["cat"])))}
    dom = {**dom, "num": nums, "cat": cats}
    target, rs = dom["target"], rng.choice([0, 7, 42, 123, 2024])
    picks = rng.sample(MODEL_POOL, 2)
    dest.mkdir(parents=True, exist_ok=True)
    _write_data(dest / "data.csv", dom, rng, rng.choice([400, 500, 600]))

    imports = sorted({p[2] for p in picks})
    models = "\n".join(f'        "{p[0]}": {p[3](rng, "RANDOM_STATE")},' for p in picks)
    (dest / "train.py").write_text(f'''import json
from pathlib import Path

import joblib
import pandas as pd
{chr(10).join(imports)}
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import build_pipeline

RANDOM_STATE = {rs}
TARGET = "{target}"


def _models():
    return {{
{models}
    }}


def _evaluate(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return {{
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
    }}


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size={rng.choice([0.2, 0.25])}, stratify=y, random_state=RANDOM_STATE)

    results, fitted = {{}}, {{}}
    for name, clf in _models().items():
        pipe = build_pipeline(clf).fit(X_tr, y_tr)
        results[name] = _evaluate(pipe, X_te, y_te)
        fitted[name] = pipe

    best = max(results, key=lambda n: results[n]["roc_auc"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps({{"best_model": best, "results": results}}, indent=2))
    return results[best]
''')
    (dest / "preprocessing.py").write_text('''import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("prep", prep), ("model", model)])
''')
    (dest / "predict.py").write_text('''import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    return model.predict(df).tolist()
''')
    (dest / "main.py").write_text(f'''import os

import pandas as pd

from predict import load_model, predict
from train import train


def main():
    out = os.environ.get("OUTPUT_DIR", "output")
    print("metrics:", train("data.csv", out))
    model = load_model(os.path.join(out, "model.joblib"))
    sample = pd.read_csv("data.csv").drop(columns=["{target}"]).head(3)
    print("predictions:", predict(model, sample))


if __name__ == "__main__":
    main()
''')
    (dest / "mlops.yaml").write_text(f"""project:
  name: {dom['name']}
  task: classification
data:
  path: data.csv
  target: {target}
evaluation:
  primary_metric: roc_auc
  min_threshold: 0.70
entrypoints:
  train: train.py
  predict: predict.py
""")
    (dest / "requirements.txt").write_text(REQUIREMENTS)
    return {"seed": seed, "domain": dom["name"], "target": target, "models": [p[0] for p in picks], "random_state": rs,
            "n_numeric": len(nums), "n_categorical": len(cats)}
