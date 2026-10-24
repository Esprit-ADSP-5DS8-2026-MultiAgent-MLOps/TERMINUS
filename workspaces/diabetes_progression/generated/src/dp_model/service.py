"""Service d'inference : charge le modele + l'ordre des features, puis predit."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_DIR = os.environ.get("DP_MODEL_DIR", "model")


def load_artifact(model_dir=DEFAULT_DIR):
    """Retourne un 'bundle' {model, features}, pas le modele seul."""
    d = Path(model_dir)
    model = joblib.load(d / "regressor.joblib")
    features = json.loads((d / "features.json").read_text(encoding="utf-8"))["features"]
    return {"model": model, "features": features}


def score(artifact, rows):
    """rows : dict ou liste de dict -> liste de float (progression de la maladie)."""
    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)[artifact["features"]]
    return [float(v) for v in artifact["model"].predict(df)]
