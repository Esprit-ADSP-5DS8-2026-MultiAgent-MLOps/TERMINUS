import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 7


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=RANDOM_STATE)
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500, random_state=RANDOM_STATE)).fit(X_tr, y_tr)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    return {"accuracy": float(accuracy_score(y_te, pipe.predict(X_te))), "roc_auc": float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))}
