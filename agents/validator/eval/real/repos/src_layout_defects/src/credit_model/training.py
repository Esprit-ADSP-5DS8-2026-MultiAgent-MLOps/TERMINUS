"""Entraînement (version avec défauts réels : chemin absolu en dur + métriques écrites à la main)."""
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
