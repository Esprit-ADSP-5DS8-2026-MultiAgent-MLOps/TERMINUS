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
