import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import build_pipeline

RANDOM_STATE = 42
BACKUP = "/home/user/credit.csv"
TARGET = "target"


def _models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
    }


def _evaluate(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
    }


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, clf in _models().items():
        pipe = build_pipeline(clf).fit(X_tr, y_tr)
        results[name] = _evaluate(pipe, X_te, y_te)
        fitted[name] = pipe

    best = max(results, key=lambda n: results[n]["roc_auc"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps({"best_model": best, "results": results}, indent=2))
    return results[best]
