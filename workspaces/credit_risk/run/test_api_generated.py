"""Tests generes automatiquement a partir du rapport d'analyse. Ne pas editer."""
import importlib
import json
import os
import sys

import httpx
import pytest

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
RUN_DIR = os.environ.get("RUN_DIR")  # present seulement en mode local

EXAMPLE_INPUT = {'status': '... < 100 DM', 'duration': 6, 'credit_history': 'critical account/other credits existing', 'purpose': 'domestic appliances', 'amount': 1169, 'savings': 'unknown/no savings account', 'employment_duration': '... >= 7 years', 'installment_rate': 4, 'personal_status_sex': 'male : single', 'other_debtors': 'none', 'present_residence': 4, 'property': 'real estate', 'age': 67, 'other_installment_plans': 'none', 'housing': 'own', 'number_credits': 2, 'job': 'skilled employee/official', 'people_liable': 1, 'telephone': 'yes', 'foreign_worker': 'yes'}
EXAMPLE_OUTPUT = {'prediction': 1, 'proba': 0.785}
FIELD_TYPES = {'status': 'str', 'duration': 'int', 'credit_history': 'str', 'purpose': 'str', 'amount': 'int', 'savings': 'str', 'employment_duration': 'str', 'installment_rate': 'int', 'personal_status_sex': 'str', 'other_debtors': 'str', 'present_residence': 'int', 'property': 'str', 'age': 'int', 'other_installment_plans': 'str', 'housing': 'str', 'number_credits': 'int', 'job': 'str', 'people_liable': 'int', 'telephone': 'str', 'foreign_worker': 'str'}
REQUIRED_FIELDS = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker']
OUTPUT_FIELDS = ['prediction']
DATA_FILE = 'data.csv'
PREDICT_MODULE = 'predict'
LOADER = 'load_model'
PREDICT_FN = 'predict'
MODEL_PATH = 'output/model.joblib'
CUSTOM_MODULE = 'preprocessing'
SUPPORTS_PROBA = True
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
