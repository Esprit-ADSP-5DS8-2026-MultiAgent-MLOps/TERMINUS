import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df[["a", "b"]], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    clf = LogisticRegression().fit(X_tr, y_tr)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, Path(output_dir) / "model.joblib")
    return {"roc_auc": float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))}
