"""API generee automatiquement. Ne pas editer a la main : regeneree par l'agent."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from classifier import TicketClassifier

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/ticket_model.joblib")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["classifier"] = TicketClassifier(model_path=MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="ticket-classifier", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "I was charged twice on my last invoice"}})
    text: str


class PredictResponse(BaseModel):
    label: str
    scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "classifier" in state}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest):
    """Route synchrone : l'inference est bloquante (CPU), FastAPI l'execute dans un thread."""
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    result = classifier.predict(req.text)
    return result
