"""Reentraine le modele (OPTIONNEL : un modele est deja fourni dans model/)."""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def main():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "regressor.joblib")
    meta = {"features": list(X.columns), "target": "disease_progression"}
    (MODEL_DIR / "features.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("R2 train:", round(model.score(X, y), 3))


if __name__ == "__main__":
    main()
