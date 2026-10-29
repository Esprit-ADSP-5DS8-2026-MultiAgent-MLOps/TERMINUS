import joblib
import pandas as pd

import preprocessing  # noqa: F401  (classes personnalisées nécessaires au chargement)


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    return model.predict(df).tolist()
