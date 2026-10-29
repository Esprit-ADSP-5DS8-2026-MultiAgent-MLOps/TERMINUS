"""Chargement et prédiction."""
import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    df = df[list(model.feature_names_in_)]
    return [int(v) for v in model.predict(df)]
