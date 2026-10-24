"""
Monitoring gateway : un proxy placé DEVANT l'API du modèle (Agent 3).

    client ──► gateway :8100 ──► API du modèle :8000
                  │
                  ├─ enregistre entrées + prédiction + latence (SQLite)
                  ├─ /metrics  pour Prometheus
                  ├─ /feedback pour recevoir la vérité terrain plus tard
                  ├─ /alerts   webhook d'Alertmanager (l'agent est prévenu des alertes)
                  └─ moniteur en tâche de fond : fenêtres -> analyse -> métriques -> agent

Pourquoi un proxy plutôt que modifier l'API générée : on ne touche pas au code de l'Agent 3,
ça marche pour n'importe quelle API qui expose /predict, et on peut le retirer sans rien casser.
La réponse du modèle est renvoyée telle quelle ; on ajoute seulement l'en-tête X-Request-ID,
que le client réutilise pour envoyer le label plus tard.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from .metrics import GatewayMetrics
from .monitor import Monitor
from .store import Store
from .target import MonitoringTarget

log = logging.getLogger("monitoring_agent.gateway")


class Feedback(BaseModel):
    request_id: str
    label: float


class FeedbackBatch(BaseModel):
    items: list[Feedback]


def create_app(target: MonitoringTarget, check_interval_s: float = 2.0,
               transport: httpx.AsyncBaseTransport | None = None, run_monitor: bool = True) -> FastAPI:
    """`transport` permet aux tests de brancher une fausse API de modèle, sans réseau."""
    store = Store(target.store_path)
    metrics = GatewayMetrics(target.project)
    monitor = Monitor(target, store, metrics)

    async def monitor_loop(client: httpx.AsyncClient) -> None:
        while True:
            try:
                r = await client.get("/health", timeout=2.0)
                metrics.upstream_up.labels(target.project).set(int(r.status_code == 200))
            except httpx.HTTPError:
                metrics.upstream_up.labels(target.project).set(0)
            try:
                await asyncio.to_thread(monitor.step)
            except Exception:  # le moniteur ne doit jamais faire tomber la gateway
                log.exception("échec d'un passage du moniteur")
            await asyncio.sleep(check_interval_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(base_url=target.upstream_url, timeout=10.0, transport=transport)
        task = asyncio.create_task(monitor_loop(app.state.client)) if run_monitor else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await app.state.client.aclose()
        store.close()

    app = FastAPI(title=f"Monitoring gateway — {target.project}", version="0.2", lifespan=lifespan)
    app.state.store, app.state.metrics, app.state.monitor, app.state.target = store, metrics, monitor, target

    @app.post("/predict")
    async def predict(request: Request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        try:
            features = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="corps JSON invalide") from None
        if not isinstance(features, dict):
            raise HTTPException(status_code=400, detail="un objet JSON par requête est attendu")

        prediction = proba = None
        try:
            upstream = await app.state.client.post("/predict", json=features)
            status, body = upstream.status_code, upstream.json()
            error = None if upstream.is_success else str(body)[:500]
            if upstream.is_success:
                prediction = body.get(target.response_prediction_field)
                proba = body.get(target.response_proba_field) if target.response_proba_field else None
        except httpx.HTTPError as exc:
            status, body, error = 502, {"detail": f"API du modèle injoignable : {type(exc).__name__}"}, str(exc)
        except ValueError:
            status, body, error = 502, {"detail": "réponse non JSON de l'API du modèle"}, "invalid json"

        elapsed = time.perf_counter() - t0
        store.add_prediction(request_id, features, prediction, proba, elapsed * 1000, status, error)
        metrics.observe_request("/predict", status, elapsed)
        if prediction is not None:
            metrics.observe_prediction(prediction, proba)
        return JSONResponse(body, status_code=status, headers={"X-Request-ID": request_id})

    @app.post("/feedback")
    async def feedback(batch: FeedbackBatch | Feedback):
        items = batch.items if isinstance(batch, FeedbackBatch) else [batch]
        matched = store.add_labels([(i.request_id, i.label) for i in items])
        metrics.labels.labels(target.project).inc(len(items))
        return {"received": len(items), "matched": matched}

    @app.post("/alerts")
    async def alerts(payload: dict):
        """Webhook Alertmanager : chaque alerte (déclenchée ou résolue) est gardée pour l'agent."""
        for a in payload.get("alerts", []):
            labels = a.get("labels", {})
            store.add_alert(a.get("status", "firing"), labels.get("alertname", "?"), labels.get("severity"), a)
        return {"stored": len(payload.get("alerts", []))}

    @app.get("/health")
    async def health():
        try:
            up = (await app.state.client.get("/health", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            up = False
        return {"status": "ok", "upstream": up, "project": target.project}

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/monitoring/status")
    async def status():
        last = monitor.last_report
        return {"project": target.project, "window_size": target.window_size, **store.traffic_stats(),
                "windows": store.last_window_idx() + 1,
                "last_window": last.model_dump(mode="json", include={"window", "severity", "trigger_agent", "signals"})
                if last else None}

    @app.get("/monitoring/windows")
    async def windows(limit: int = 100):
        return store.window_reports(limit)

    @app.get("/monitoring/alerts")
    async def recent_alerts(limit: int = 50):
        return store.recent_alerts(limit)

    return app
