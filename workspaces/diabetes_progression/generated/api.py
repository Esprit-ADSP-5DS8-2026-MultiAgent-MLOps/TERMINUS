"""API generee automatiquement. Ne pas editer a la main : regeneree par l'agent."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.dp_model.service import load_artifact, score

MODEL_PATH = os.environ.get("MODEL_PATH", "model")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_artifact(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="diabetes-progression", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"age": 0.038076, "bmi": 0.061696, "bp": 0.021872, "s1": -0.044223, "s2": -0.034821, "s3": -0.043401, "s4": -0.002592, "s5": 0.019907, "s6": -0.017646, "sex": 0.05068}})
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class PredictResponse(BaseModel):
    prediction: float


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
    pred = score(model, data)[0]
    out = {"prediction": round(float(pred), 1)}
    return out
