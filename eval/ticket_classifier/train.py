import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train(data_path="data/tickets.csv", output_dir="artifacts"):
    df = pd.read_csv(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X_tr, y_tr)
    acc = float(accuracy_score(y_te, pipe.predict(X_te)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "ticket_model.joblib")
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))
    return acc


if __name__ == "__main__":
    print("accuracy:", train())
