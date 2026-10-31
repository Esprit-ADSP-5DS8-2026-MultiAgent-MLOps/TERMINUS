"""Expérience B : régression logistique."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["default"]), df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    m = LogisticRegression(max_iter=500, random_state=1).fit(X_tr, y_tr)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(m, Path(output_dir) / "model.joblib")
    return {"accuracy": float(m.score(X_te, y_te))}
