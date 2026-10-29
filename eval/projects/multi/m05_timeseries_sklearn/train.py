from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 3
FEATURES = ["t", "season"]


def add_features(df):
    df = df.copy()
    df["season"] = df["t"] % 12
    return df[FEATURES]


def train(data_path, output_dir):
    s = pd.read_csv(data_path).sort_values("t").reset_index(drop=True)
    cut = int(len(s) * 0.8)                                  # découpage chronologique : pas de fuite du futur
    X, y = add_features(s), s["y"]
    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE).fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    return {"r2": float(r2_score(y.iloc[cut:], pred)), "rmse": float(mean_squared_error(y.iloc[cut:], pred) ** 0.5)}
