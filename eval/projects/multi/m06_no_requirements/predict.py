import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data) if isinstance(data, list) else data
    return model.predict(df).tolist()
