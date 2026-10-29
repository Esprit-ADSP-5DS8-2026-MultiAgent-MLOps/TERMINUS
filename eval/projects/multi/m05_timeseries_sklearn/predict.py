import joblib
import pandas as pd

from train import add_features


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data) if isinstance(data, list) else data
    return model.predict(add_features(df)).tolist()
