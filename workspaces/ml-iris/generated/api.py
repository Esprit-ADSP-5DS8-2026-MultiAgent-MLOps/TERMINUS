"""API generee automatiquement. Ne pas editer a la main : regeneree par l'agent."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

import joblib

# main.py is not imported to avoid multipart dependency
# The model is loaded in the lifespan context and used directly in predict_route


MODEL_PATH = os.environ.get("MODEL_PATH", "iris_model.pkl")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="ml-iris-pipeline", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"petal_length": 1.4, "petal_width": 0.2, "sepal_length": 5.1, "sepal_width": 3.5}})
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictResponse(BaseModel):
    flower_type: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest):
    """Route synchrone : l'inference est bloquante (CPU), FastAPI l'exécute dans un thread."""
    # Use the model loaded in the lifespan context
    data_list = [req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]
    model = state.get("model")
    if model is None:
        raise RuntimeError("Model not loaded")
    prediction = model.predict([data_list])[0]
    flower_names = ['Setosa', 'Versicolor', 'Virginica']
    return {"flower_type": flower_names[int(prediction)]}

