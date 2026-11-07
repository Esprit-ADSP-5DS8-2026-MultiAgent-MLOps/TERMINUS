import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocessing import build_pipeline

RANDOM_STATE = 42
TARGET = "price_k"


def _models():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE),
    }


def _evaluate(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
    }


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, reg in _models().items():
        pipe = build_pipeline(reg).fit(X_tr, y_tr)
        results[name] = _evaluate(pipe, X_te, y_te)
        fitted[name] = pipe

    best = min(results, key=lambda n: results[n]["rmse"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps({"best_model": best, "results": results}, indent=2))
    return results[best]
