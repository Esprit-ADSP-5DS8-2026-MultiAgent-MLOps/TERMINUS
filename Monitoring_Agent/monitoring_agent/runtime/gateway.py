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
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from .metrics import GatewayMetrics
from .monitor import Monitor
from .store import Store
from .supervisor import Supervisor
from .target import MonitoringTarget

log = logging.getLogger("monitoring_agent.gateway")


class Feedback(BaseModel):
    request_id: str
    label: float


class FeedbackBatch(BaseModel):
    items: list[Feedback]


class HumanAnswer(BaseModel):
    by: str = "humain"
    comment: str = ""


class SimulationRequest(BaseModel):
    scenario: str = "covariate"
    n: int = 1200
    warmup: int = 400
    rate: float = 20.0
    label_delay: int = 200


def create_app(target: MonitoringTarget, check_interval_s: float = 2.0,
               transport: httpx.AsyncBaseTransport | None = None, run_monitor: bool = True,
               supervisor: Supervisor | None = None, agent: bool = True) -> FastAPI:
    """`transport` permet aux tests de brancher une fausse API de modèle ; `supervisor` un agent de test."""
    store = Store(target.store_path)
    metrics = GatewayMetrics(target.project)
    if agent and supervisor is None:
        supervisor = Supervisor(target, store)
    monitor = Monitor(target, store, metrics, on_trigger=supervisor.on_trigger if supervisor else None)
    # Serveur MCP du projet, monté sur /mcp/ (Streamable HTTP) : Claude Desktop, MCP Inspector, l'orchestrateur
    # ou un autre agent peuvent interroger le monitoring avec les mêmes outils que l'enquêteur.
    from ..agent.mcp_server import server_for_target
    mcp_server = server_for_target(target, store)
    from mcp.server.transport_security import TransportSecuritySettings
    hosts = os.environ.get("MONITORING_MCP_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*,host.docker.internal:*").split(",")
    mcp_app = mcp_server.streamable_http_app(   # protection DNS-rebinding : seuls ces hôtes sont acceptés
        streamable_http_path="/", transport_security=TransportSecuritySettings(allowed_hosts=hosts))

    async def monitor_loop(client: httpx.AsyncClient) -> None:
        while True:
            try:
                r = await client.get("/health", timeout=2.0)
                up = r.status_code == 200
            except httpx.HTTPError:
                up = False
            metrics.upstream_up.labels(target.project).set(int(up))
            app.state.upstream_up = up
            try:
                await asyncio.to_thread(monitor.step)
            except Exception:  # le moniteur ne doit jamais faire tomber la gateway
                log.exception("échec d'un passage du moniteur")
            await asyncio.sleep(check_interval_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(base_url=target.upstream_url, timeout=10.0, transport=transport)
        task = asyncio.create_task(monitor_loop(app.state.client)) if run_monitor else None
        async with mcp_server.session_manager.run():   # le cycle de vie d'une sous-app montée n'est pas lancé seul
            yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await app.state.client.aclose()
        if supervisor:
            supervisor.close()
        store.close()

    app = FastAPI(title=f"Monitoring gateway — {target.project}", version="0.2", lifespan=lifespan)
    app.state.store, app.state.metrics, app.state.monitor, app.state.target = store, metrics, monitor, target
    app.state.supervisor = supervisor
    app.mount("/mcp", mcp_app)

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

    app.state.simulation = {"running": False}
    app.state.upstream_up = None

    @app.get("/monitoring/status")
    async def status():
        last = monitor.last_report
        agent_info = None
        if supervisor:
            a = supervisor.agent
            agent_info = {"decider": a.decider.name if a.decider else "règles (aucune clé LLM)",
                          "judge": a.judge.name if a.judge else None,
                          "busy": bool(supervisor.running and not supervisor.running.done()),
                          "busy_window": supervisor.busy_window,
                          "pending_window": supervisor.pending.window.index if supervisor.pending else None}
        m = target.mapping
        return {"project": target.project, "window_size": target.window_size, **store.traffic_stats(),
                "windows": store.last_window_idx() + 1, "upstream_url": target.upstream_url,
                "upstream_up": app.state.upstream_up, "task": m.task, "metric": m.primary_metric,
                "min_threshold": m.min_threshold, "features": m.features, "model_uri": target.model_uri,
                "agent": agent_info, "simulation": app.state.simulation,
                "last_window": last.model_dump(mode="json", include={"window", "severity", "trigger_agent", "signals"})
                if last else None}

    @app.post("/monitoring/simulate")
    async def start_simulation(req: SimulationRequest, request: Request):
        """Lance du trafic simulé contre CETTE gateway (démo depuis la console)."""
        import pandas as pd

        from .simulator import SCENARIOS, simulate

        if req.scenario not in SCENARIOS:
            raise HTTPException(status_code=422, detail=f"scénario inconnu ({', '.join(SCENARIOS)})")
        if app.state.simulation.get("running"):
            raise HTTPException(status_code=409, detail="une simulation est déjà en cours")
        data = pd.read_parquet(target.reference_path).drop(columns=["y_pred", "y_pred_proba"], errors="ignore")
        sim = app.state.simulation = {"running": True, "scenario": req.scenario, "total": req.n, "sent": 0,
                                      "errors": 0, "labels": 0}

        def progress(res):
            sim.update(sent=res.sent, errors=res.errors, labels=res.labels_sent)

        async def run():
            try:
                res = await simulate(str(request.base_url).rstrip("/"), data, target.mapping.target, req.scenario,
                                     req.n, req.warmup, req.rate, req.label_delay, final_flush=True,
                                     on_progress=progress)
                progress(res)
            except Exception as exc:  # noqa: BLE001
                sim["error"] = str(exc)
            finally:
                sim["running"] = False

        app.state.sim_task = asyncio.create_task(run())
        return sim

    @app.get("/monitoring/decisions/{decision_id}/report")
    async def decision_report(decision_id: str):
        d = store.decision(decision_id)
        path = (d or {}).get("outputs", {}).get("incident")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail="pas de rapport d'incident pour cette décision")
        return Response(Path(path).read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")

    console_dir = Path(__file__).resolve().parent.parent / "console"
    if console_dir.exists():
        app.mount("/console", StaticFiles(directory=console_dir, html=True), name="console")

        @app.get("/", include_in_schema=False)
        async def home():
            return RedirectResponse("/console/")

    @app.get("/monitoring/windows")
    async def windows(limit: int = 100):
        return store.window_reports(limit)

    @app.get("/monitoring/alerts")
    async def recent_alerts(limit: int = 50, firing: bool = False):
        return store.firing_alerts() if firing else store.recent_alerts(limit)

    # -- décisions de l'agent ------------------------------------------------------------

    def _need_agent() -> Supervisor:
        if supervisor is None:
            raise HTTPException(status_code=503, detail="agent désactivé (gateway lancée avec --no-agent)")
        return supervisor

    @app.get("/monitoring/decisions")
    async def decisions(limit: int = 50):
        return store.decisions(limit)

    @app.get("/monitoring/decisions/{decision_id}")
    async def decision(decision_id: str):
        d = store.decision(decision_id)
        if d is None:
            raise HTTPException(status_code=404, detail="décision inconnue")
        return d

    @app.post("/monitoring/decide")
    async def decide_now():
        """Force l'agent sur la dernière fenêtre (démo, ou après une correction)."""
        sup = _need_agent()
        try:
            run = await asyncio.to_thread(sup.decide_now)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return run.model_dump(mode="json")

    @app.post("/monitoring/decisions/{decision_id}/{answer}")
    async def answer_decision(decision_id: str, answer: str, body: HumanAnswer | None = None):
        if answer not in ("approve", "reject"):
            raise HTTPException(status_code=404, detail="réponse attendue : approve ou reject")
        sup, body = _need_agent(), body or HumanAnswer()
        try:
            run = await asyncio.to_thread(sup.resume, decision_id, answer, body.by, body.comment)
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return run.model_dump(mode="json")

    return app
