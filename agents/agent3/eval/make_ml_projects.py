"""Genere deux projets ML de test dans eval/projects/ (deterministe, versions epinglees).

  diabetes_progression : regression sklearn, modele DEJA FOURNI, layout src/, fonctions
                         non standard (load_artifact / score), artefact + fichier de features.
  ticket_classifier    : classification de TEXTE (TF-IDF + LogisticRegression), modele
                         NON fourni (train.py), inference via une CLASSE (TicketClassifier).

Usage : python eval/make_ml_projects.py      (depuis la racine du depot, venv actif)
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "projects"
REQ = "scikit-learn==1.8.0\npandas==3.0.2\nnumpy==2.4.4\njoblib==1.5.3\n"


def write(project: str, rel: str, content: str) -> None:
    path = ROOT / project / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ------------------------------------------------------------------ diabetes_progression
SERVICE = '''\
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
'''

RETRAIN = '''\
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
'''

DP_EXAMPLE = {"age": 0.038076, "sex": 0.05068, "bmi": 0.061696, "bp": 0.021872, "s1": -0.044223,
              "s2": -0.034821, "s3": -0.043401, "s4": -0.002592, "s5": 0.019907, "s6": -0.017646}


def build_diabetes() -> None:
    p = "diabetes_progression"
    write(p, "requirements.txt", REQ)
    write(p, "src/dp_model/__init__.py", "")
    write(p, "src/dp_model/service.py", SERVICE)
    write(p, "scripts/retrain.py", RETRAIN)
    subprocess.run([sys.executable, "scripts/retrain.py"], cwd=ROOT / p, check=True)

    import joblib
    import pandas as pd
    model = joblib.load(ROOT / p / "model" / "regressor.joblib")
    expected = round(float(model.predict(pd.DataFrame([DP_EXAMPLE]))[0]), 1)
    write(p, "tests/test_service.py", f'''\
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dp_model.service import load_artifact, score  # noqa: E402

EXAMPLE = {DP_EXAMPLE!r}
EXPECTED = {expected}  # progression de la maladie predite pour EXAMPLE


def test_score_example():
    artifact = load_artifact(Path(__file__).resolve().parents[1] / "model")
    assert score(artifact, EXAMPLE) == [pytest.approx(EXPECTED, abs=0.5)]
''')


# ------------------------------------------------------------------ ticket_classifier
TRAIN = '''\
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
'''

CLASSIFIER = '''\
import os

import joblib


class TicketClassifier:
    """Classe d'inference : le modele est charge une fois, dans le constructeur."""

    def __init__(self, model_path=None):
        self.model_path = model_path or os.environ.get("MODEL_PATH", "artifacts/ticket_model.joblib")
        self.model = joblib.load(self.model_path)

    def predict(self, text):
        probs = self.model.predict_proba([text])[0]
        scores = {c: float(p) for c, p in zip(self.model.classes_, probs)}
        label = max(scores, key=scores.get)
        return {"label": label, "scores": scores}
'''

README = "# ticket_classifier\n\nClasse des tickets de support (billing, technical, account).\n\n    python train.py          # entraine et ecrit artifacts/ticket_model.joblib\n"

VOCAB = {
    "billing": (["invoice", "payment", "refund", "charge", "subscription", "credit card", "receipt", "billing address"],
                ["was charged twice", "is incorrect", "has not arrived", "failed to process",
                 "needs to be refunded", "shows the wrong amount", "was declined"]),
    "technical": (["app", "website", "server", "login page", "upload", "export", "dashboard", "API"],
                  ["keeps crashing", "returns an error", "is very slow", "does not load",
                   "shows a blank screen", "times out", "fails after the update"]),
    "account": (["password", "email address", "account", "profile", "two-factor code", "username", "settings"],
                ["cannot be reset", "needs to be changed", "was locked", "is not recognized",
                 "must be deleted", "cannot be verified"]),
}
OPENERS = ["Hello,", "Hi team,", "Please help:", "Urgent:", "", "Good morning,"]
CLOSERS = ["Thanks.", "Can you check?", "This is the second time.", "", "Please advise."]


def build_tickets() -> None:
    p = "ticket_classifier"
    rng = random.Random(42)
    rows = ["text,label"]
    for label, (subjects, issues) in VOCAB.items():
        for _ in range(100):
            text = f"{rng.choice(OPENERS)} my {rng.choice(subjects)} {rng.choice(issues)}. {rng.choice(CLOSERS)}"
            rows.append(f'"{" ".join(text.split())}",{label}')
    header, body = rows[0], rows[1:]
    rng.shuffle(body)
    write(p, "data/tickets.csv", "\n".join([header] + body) + "\n")
    write(p, "requirements.txt", REQ)
    write(p, "train.py", TRAIN)
    write(p, "classifier.py", CLASSIFIER)
    write(p, "README.md", README)


if __name__ == "__main__":
    build_diabetes()
    build_tickets()
    print("Projets crees dans", ROOT)
