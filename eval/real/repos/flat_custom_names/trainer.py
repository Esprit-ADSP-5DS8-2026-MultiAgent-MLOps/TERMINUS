"""Entraînement : module à la racine, fonction nommée fit()."""
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
