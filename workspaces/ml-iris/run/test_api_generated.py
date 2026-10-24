"""Tests generes automatiquement a partir du rapport d'analyse. Ne pas editer."""
import importlib
import json
import os
import sys

import httpx
import pytest

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
RUN_DIR = os.environ.get("RUN_DIR")  # present seulement en mode local

EXAMPLE_INPUT = {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}
EXAMPLE_OUTPUT = {'flower_type': 'Setosa'}
FIELD_TYPES = {'sepal_length': 'float', 'sepal_width': 'float', 'petal_length': 'float', 'petal_width': 'float'}
REQUIRED_FIELDS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
OUTPUT_FIELDS = ['flower_type']
DATA_FILE = None
PREDICT_MODULE = 'main'
LOADER = None
PREDICT_FN = 'predict'
MODEL_PATH = 'iris_model.pkl'
CUSTOM_MODULE = None
SUPPORTS_PROBA = False
N_CONSISTENCY_ROWS = 20

BAD_VALUES = {"int": "abc", "float": "abc", "bool": "maybe", "str": [1, 2]}


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("model_loaded") is True


def test_predict_example(client):
    r = client.post("/predict", json=EXAMPLE_INPUT)
    assert r.status_code == 200, r.text
    body = r.json()
    for name in OUTPUT_FIELDS:
        assert name in body, f"champ de sortie absent: {name}"
    if EXAMPLE_OUTPUT:
        assert body["prediction"] == EXAMPLE_OUTPUT["prediction"]
        if "proba" in EXAMPLE_OUTPUT and body.get("proba") is not None:
            assert body["proba"] == pytest.approx(EXAMPLE_OUTPUT["proba"], abs=0.02)


@pytest.mark.parametrize("missing", REQUIRED_FIELDS)
def test_missing_field_returns_422(client, missing):
    payload = {k: v for k, v in EXAMPLE_INPUT.items() if k != missing}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("name", list(FIELD_TYPES))
def test_wrong_type_returns_422(client, name):
    payload = dict(EXAMPLE_INPUT)
    payload[name] = BAD_VALUES[FIELD_TYPES[name]]
    assert client.post("/predict", json=payload).status_code == 422


def test_openapi_exposes_input_fields(client):
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    exposed = set()
    for s in schemas.values():
        exposed |= set(s.get("properties", {}))
    assert set(FIELD_TYPES) <= exposed


def test_api_matches_direct_model_call(client):
    """L'API doit donner EXACTEMENT les memes predictions que le modele appele directement."""
    if not RUN_DIR or not DATA_FILE:
        pytest.skip("RUN_DIR ou fichier de donnees absent")
    import pandas as pd

    sys.path.insert(0, RUN_DIR)
    if CUSTOM_MODULE:
        importlib.import_module(CUSTOM_MODULE)
    mod = importlib.import_module(PREDICT_MODULE)
    load = getattr(mod, LOADER) if LOADER else importlib.import_module("joblib").load
    model = load(os.path.join(RUN_DIR, MODEL_PATH))
    df = pd.read_csv(os.path.join(RUN_DIR, DATA_FILE))[list(FIELD_TYPES)].head(N_CONSISTENCY_ROWS)
    direct = getattr(mod, PREDICT_FN)(model, df)
    rows = json.loads(df.to_json(orient="records"))
    for i, row in enumerate(rows):
        r = client.post("/predict", json=row)
        assert r.status_code == 200, f"ligne {i}: {r.text}"
        assert r.json()["prediction"] == direct[i], f"ligne {i}: API != modele"
        if SUPPORTS_PROBA and r.json().get("proba") is not None:
            p = float(model.predict_proba(df.iloc[[i]])[0, 1])
            assert r.json()["proba"] == pytest.approx(p, abs=1e-6), f"ligne {i}: proba differente"
