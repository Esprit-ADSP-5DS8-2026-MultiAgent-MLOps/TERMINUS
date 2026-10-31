#!/usr/bin/env python
"""Génère eval/projects/multi/ : projets PyTorch, TensorFlow, NLP, vision, séries temporelles, sans requirements.txt, pyproject seul,
notebook seul, et deux projets « entraînement seul ». Petits (< 20 Ko), déterministes. Annotations -> eval/multi_expected.json.
    python eval/make_multi.py
Ces projets sont marqués `holdout` : à ne PAS utiliser pour concevoir le prompt ni le contrat.
"""
import csv
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "projects" / "multi"
SK = "scikit-learn==1.8.0\npandas==3.0.2\nnumpy==2.4.4\njoblib==1.5.3\n"

MAIN = '''import os

import pandas as pd

from predict import load_model, predict
from train import train


def main():
    out = os.environ.get("OUTPUT_DIR", "output")
    print("metrics:", train("data.csv", out))
    model = load_model(os.path.join(out, "model.joblib"))
    print("predictions:", predict(model, pd.read_csv("data.csv").drop(columns=["{t}"]).head(3)))


if __name__ == "__main__":
    main()
'''
PREDICT = '''import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data) if isinstance(data, list) else data
    return model.predict(df).tolist()
'''
MLOPS = "project:\n  name: {n}\n  task: {task}\ndata:\n  path: data.csv\n  target: {t}\nevaluation:\n  primary_metric: {m}\n  min_threshold: {th}\nentrypoints:\n  train: train.py\n  predict: predict.py\n"
TRAIN_TEXT = '''import json
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
'''
PREDICT_TEXT = PREDICT.replace("model.predict(df)", "model.predict(df['text'] if hasattr(df, 'columns') and 'text' in df else df)")
TRAIN_TS = '''from pathlib import Path

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
'''
PREDICT_TS = '''import joblib
import pandas as pd

from train import add_features


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data) if isinstance(data, list) else data
    return model.predict(add_features(df)).tolist()
'''
TRAIN_TRAINONLY = '''from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train(data_path, output_dir):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "model.joblib")
    return {"roc_auc": float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))}
'''
TRAIN_TORCH = '''import torch
from torch import nn


def train(data_path, output_dir):
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x, y = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    loss = nn.functional.cross_entropy(net(x), y)
    loss.backward()
    opt.step()
    torch.save(net.state_dict(), output_dir + "/model.pt")
    return {"accuracy": 0.1}
'''
TRAIN_TF = '''import tensorflow as tf


def train(data_path, output_dir):
    model = tf.keras.Sequential([tf.keras.layers.TextVectorization(), tf.keras.layers.Dense(1)])
    model.compile(loss="mse")
    return {"accuracy": 0.5}
'''


def w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_csv(p: Path, header, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])


def tabular_rows(n=200, seed=1):
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = r.gauss(0, 1), r.gauss(0, 1)
        rows.append([round(a, 4), round(b, 4), int(a + 0.5 * b + r.gauss(0, 0.5) > 0)])
    return rows


def text_rows(n=200, seed=2):
    r = random.Random(seed)
    pos, neg = "great good love excellent happy fine".split(), "bad awful hate poor sad broken".split()
    rows = []
    for _ in range(n):
        lab = r.randint(0, 1)
        rows.append([" ".join(r.choice(pos if lab else neg) if r.random() < 0.8 else r.choice(pos + neg) for _ in range(6)), lab])
    return rows


def ts_rows(n=200, seed=3):
    r = random.Random(seed)
    v, rows = 0.0, []
    for i in range(n):
        v = 0.8 * v + r.gauss(0, 1) + (1 if i % 12 == 0 else 0)
        rows.append([i, round(v, 4)])
    return rows


def sk_project(d: Path, name, header, rows, target, task="classification", metric="roc_auc", th=0.6, train=None, predict=PREDICT, requirements=SK, extra=True):
    write_csv(d / "data.csv", header, rows)
    w(d / "train.py", train)
    if extra:
        w(d / "predict.py", predict)
        w(d / "main.py", MAIN.replace("{t}", target))
        w(d / "mlops.yaml", MLOPS.format(n=name, task=task, t=target, m=metric, th=th))
    if requirements:
        w(d / "requirements.txt", requirements)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    exp = {}

    def case(name, expected, profile="sklearn-tabular", framework="scikit-learn", note=""):
        exp[name] = {"path": f"projects/multi/{name}", "profile": profile, "framework": framework, "expected_checks": expected, "note": note}

    # --- hors périmètre (frameworks lourds) : REJECTED, jamais réparés ---
    for name, train, req, fw, kind in (("m01_pytorch_mnist", TRAIN_TORCH, "torch==2.3.0\n", "pytorch", "vision"),
                                       ("m02_tensorflow_text", TRAIN_TF, "tensorflow==2.16.1\n", "tensorflow", "nlp"),
                                       ("m04_vision_torchvision", TRAIN_TORCH.replace("import torch", "import torch\nimport torchvision  # noqa"), "torch==2.3.0\ntorchvision==0.18.0\n", "pytorch", "vision")):
        sk_project(OUT / name, name, ["a", "b", "target"], tabular_rows(20), "target", train=train, requirements=req)
        case(name, ["FRAMEWORK.SCOPE"], framework=fw, note=f"{kind} : hors périmètre V1 (déclaré, jamais exécuté)")
    # --- conformes ---
    sk_project(OUT / "m03_nlp_tfidf_sklearn", "nlp-tfidf", ["text", "label"], text_rows(), "label", train=TRAIN_TEXT, predict=PREDICT_TEXT)
    case("m03_nlp_tfidf_sklearn", [], note="NLP scikit-learn (TF-IDF) : conforme")
    sk_project(OUT / "m05_timeseries_sklearn", "ts-lags", ["t", "y"], ts_rows(), "y", task="regression", metric="r2", th=-5.0, train=TRAIN_TS, predict=PREDICT_TS)
    case("m05_timeseries_sklearn", [], note="série temporelle, découpage chronologique : conforme")
    # --- environnement non standard ---
    sk_project(OUT / "m06_no_requirements", "no-req", ["a", "b", "target"], tabular_rows(), "target", train=TRAIN_TRAINONLY, requirements=None)
    case("m06_no_requirements", ["STRUCT.REQUIRED_FILES"], note="aucun fichier de dépendances : EXEC.RUN attendu NOT_CHECKED")
    sk_project(OUT / "m07_pyproject_only", "pyproject", ["a", "b", "target"], tabular_rows(), "target", train=TRAIN_TRAINONLY, requirements=None)
    w(OUT / "m07_pyproject_only" / "pyproject.toml", '[project]\nname = "x"\nversion = "0.1"\ndependencies = ["scikit-learn==1.8.0", "pandas==3.0.2", "numpy==2.4.4", "joblib==1.5.3"]\n')
    case("m07_pyproject_only", ["STRUCT.REQUIRED_FILES"], note="dépendances dans pyproject.toml : V1 exige requirements.txt, EXEC.RUN attendu NOT_CHECKED")
    d = OUT / "m08_notebook_only"
    write_csv(d / "data.csv", ["a", "b", "target"], tabular_rows(50))
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": ["import pandas as pd\n", "from sklearn.ensemble import RandomForestClassifier\n", "df = pd.read_csv('data.csv')\n", "RandomForestClassifier().fit(df[['a','b']], df['target'])\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    w(d / "analysis.ipynb", json.dumps(nb, indent=1))
    case("m08_notebook_only", ["STRUCT.REQUIRED_FILES", "ML.TRAIN_INTERFACE"], note="notebook seul, sans train.py ni requirements.txt")
    # --- profil « entraînement seul » ---
    sk_project(OUT / "t01_training_only_ok", "train-only", ["a", "b", "target"], tabular_rows(), "target", train=TRAIN_TRAINONLY, extra=False)
    case("t01_training_only_ok", [], profile="training-only", note="pas de predict.py ni main.py : conforme au profil entraînement seul")
    sk_project(OUT / "t02_training_only_unpinned", "train-only-unpinned", ["a", "b", "target"], tabular_rows(), "target", train=TRAIN_TRAINONLY, extra=False,
               requirements="scikit-learn>=1.0\npandas\nnumpy==2.4.4\njoblib==1.5.3\n")
    case("t02_training_only_unpinned", ["ENV.DEPENDENCIES_PINNED"], profile="training-only", note="dépendances non épinglées")
    (HERE / "multi_expected.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(exp)} projets -> {OUT}")


if __name__ == "__main__":
    main()
