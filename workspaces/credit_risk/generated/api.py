"""API generee automatiquement. Ne pas editer a la main : regeneree par l'agent."""
import os
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import preprocessing  # noqa: F401  (classes personnalisees requises par joblib.load)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from predict import load_model, predict

MODEL_PATH = os.environ.get("MODEL_PATH", "output/model.joblib")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="credit-risk", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"age": 67, "amount": 1169, "credit_history": "critical account/other credits existing", "duration": 6, "employment_duration": "... \u003e= 7 years", "foreign_worker": "yes", "housing": "own", "installment_rate": 4, "job": "skilled employee/official", "number_credits": 2, "other_debtors": "none", "other_installment_plans": "none", "people_liable": 1, "personal_status_sex": "male : single", "present_residence": 4, "property": "real estate", "purpose": "domestic appliances", "savings": "unknown/no savings account", "status": "... \u003c 100 DM", "telephone": "yes"}})
    status: str
    duration: int
    credit_history: str
    purpose: str
    amount: int
    savings: str
    employment_duration: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    present_residence: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    number_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str


class PredictResponse(BaseModel):
    prediction: int
    proba: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest):
    """Route synchrone : l'inference est bloquante (CPU), FastAPI l'execute dans un thread."""
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    data = req.model_dump()
    pred = predict(model, data)[0]
    out = {"prediction": int(pred)}
    out["proba"] = float(model.predict_proba(pd.DataFrame([data]))[0, 1])
    return out
