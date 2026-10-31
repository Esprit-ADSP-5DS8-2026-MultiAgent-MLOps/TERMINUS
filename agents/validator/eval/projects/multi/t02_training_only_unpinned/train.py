from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "model.joblib")
    return {"roc_auc": float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))}
