"""Entraînement du modèle de défaut de paiement."""
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
