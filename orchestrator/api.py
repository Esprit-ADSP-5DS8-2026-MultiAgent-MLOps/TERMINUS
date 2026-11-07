"""
L'API de Terminus : ce que lit et commande l'interface web (React).

    python -m orchestrator api                 # http://127.0.0.1:8090/api/...   (docs : /docs)

Elle ne contient aucune logique de pipeline : tout passe par Run (graph.py), comme la ligne de
commande et le service Slack. Un run ne s'exécute qu'à un endroit à la fois (runlock.py).

Lecture : santé des services, projets, lignes (runs) et leur plan, décisions en attente, fichiers d'un
run, registre MLflow. Commandes : lancer une ligne, décider à une validation, relancer le monitoring,
envoyer une requête au modèle en production (via la gateway de l'Agent 4).
Aucun secret n'est jamais renvoyé : pour les clés, seulement « présente » ou « absente ».
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from common.envscope import read_env_file

from .checkpoint import describe_backend, get_checkpointer
from .graph import Run, RunNotFound
from .runlock import RunBusy, is_busy, run_lock

REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOTS = [REPO / "eval" / "projects", REPO / "agents" / "validator" / "eval" / "projects" / "variants"]
FINAL = {"DONE", "BLOCKED", "FAILED", "REJECTED", "MANIFEST_PROPOSED", "CHAMPION_KEPT"}

# Plan de ligne : les gares dans l'ordre, et leur nom à l'écran
STATIONS = {
    "validator": "Validateur v1", "repair": "Réparation v2", "propose_manifest": "Manifeste (Agent 1)",
    "agent1": "Agent 1 · Analyse", "agent2": "Agent 2 · MLflow", "agent3": "Agent 3 · API Docker",
    "agent4": "Agent 4 · Monitoring",
    "gate_manifest": "Signal manifeste", "gate_champion": "Signal champion", "gate_retraining": "Signal réentraînement",
    "gate_failure": "Signal échec", "gate_repair_review": "Signal réparation",
    "gate_validator_override": "Signal validateurs",
}
GATE_NODE = {"manifest_confirmation": "gate_manifest", "champion_replacement": "gate_champion",
             "retraining": "gate_retraining", "failure_review": "gate_failure",
             "repair_review": "gate_repair_review", "validator_override": "gate_validator_override"}
CANONICAL = ["validator", "agent1", "agent2", "gate_champion", "agent3", "agent4"]


class StartRequest(BaseModel):
    project_dir: str
    supervisor: str | None = None


class Agent4Answer(BaseModel):
    decision: str                       # approve | reject
    by: str
    comment: str = ""


class DecisionRequest(BaseModel):
    decision: str                       # approve | reject
    by: str
    comment: str = ""
    key: str | None = None              # gate attendue (« champion_replacement:1 ») : protège d'un double clic
    publish: bool | None = None         # champion : publier sur Docker Hub


# ---------------------------------------------------------------------------
# Lecture d'un run
# ---------------------------------------------------------------------------


def _read(path: str | None) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}


def gate_key(run: Run) -> str | None:
    q = run.question
    return f"{q['gate']}:{len(run.state.get('decisions', []))}" if q else None


def project_of(state: dict) -> str:
    spec = _read(state.get("artifacts", {}).get("project_spec"))
    return spec.get("project_name") or Path(state.get("project_dir", "?")).name


# Équivalence approximative entre les codes du v1 (validator/) et les contrôles du v2 (agents/validator/policy) :
# sert à mettre les deux verdicts côte à côte, gare par gare.
V1_TO_V2 = [("C1.", "STRUCT.REQUIRED_FILES"), ("C12.", "STRUCT.LIMITS"), ("C8.", "STRUCT.MANIFEST"),
            ("C2.SYNTAX", "CODE.SYNTAX"), ("C2.", "FRAMEWORK.SCOPE"), ("C6.UNDECLARED", "ENV.DEPENDENCIES_DECLARED"),
            ("C6.", "ENV.DEPENDENCIES_PINNED"), ("C3.", "ML.TRAIN_INTERFACE"), ("C4.", "ML.PREDICT_INTERFACE"),
            ("C5.INTERACTIVE", "ML.NON_INTERACTIVE"), ("C5.", "EXEC.RUN"), ("C7.", "DATA.USABLE"),
            ("C9.", "ML.REPRODUCIBILITY"), ("C10.", "IO.PATHS"), ("C11.SECRET", "SEC.SECRETS"),
            ("C11.NETWORK", "SEC.NETWORK"), ("C11.COMMITTED", "SEC.COMMITTED_ARTIFACTS"), ("C11.", "SEC.DANGEROUS_CALLS")]


def _v2_check(code: str) -> str | None:
    return next((v2 for prefix, v2 in V1_TO_V2 if code.startswith(prefix)), None)


def _fmt(v, digits: int = 4) -> str:
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def compare_validators(v1: dict, v2: dict) -> dict:
    """Mesure v1/v2 d'une ligne : chaque bloquant du v1 face au contrôle équivalent du v2, puis ce que seul
    le v2 a vu. Les deux validateurs n'ont pas la même grille : l'équivalence est indicative."""
    checks = {c.get("check_id"): c.get("status") for c in v2.get("checks", [])}
    issues = v2.get("issues", [])
    rows, seen = [], set()
    for f in v1.get("findings", []):
        if f.get("severity") != "BLOCKER":
            continue
        target = _v2_check(f["code"])
        seen.add(target)
        st = checks.get(target)
        verdict = ("confirme" if st == "FAIL" else "ne confirme pas" if st in ("PASS", "WARNING")
                   else "non vérifié" if st else "pas d'équivalent")
        rows.append({"v1": f["code"], "v1_message": f.get("message", "")[:160], "v2": target, "v2_status": st,
                     "verdict": verdict})
    only_v2 = [{"v2": i.get("check_id"), "description": (i.get("description") or "")[:160],
                "severity": i.get("severity")} for i in issues if i.get("check_id") not in seen]
    agree = sum(r["verdict"] == "confirme" for r in rows)
    return {"rows": rows, "only_v2": only_v2[:8], "v1_blockers": len(rows), "confirmed": agree,
            "v2_status": v2.get("status"), "v2_next_action": v2.get("next_action"),
            "summary": (f"{agree}/{len(rows)} bloquant(s) du v1 confirmé(s) par le v2"
                        + (f", {len(only_v2)} problème(s) vu(s) seulement par le v2" if only_v2 else ""))}


def station_facts(node: str, entry: dict, ctx: dict) -> dict:
    """Ce que la gare a fait, lu dans ses fichiers de sortie. Un agent relancé (réentraînement) réécrit les
    mêmes fichiers : seule sa DERNIÈRE exécution est lue ici (voir track), les précédentes gardent leurs raisons."""
    out = entry.get("outputs") or {}
    facts: list[list[str]] = []
    extra: dict = {}
    if node == "validator":
        r = _read(out.get("compatibility_report"))
        ctx["v1"] = r
        if r:
            sm = r.get("summary", {})
            facts += [["verdict", r.get("verdict", "?")],
                      ["bloquants / avertissements", f"{sm.get('BLOCKER', 0)} / {sm.get('WARNING', 0)}"],
                      ["niveaux", " · ".join(f"{l['name']} {'✓' if l['status'] == 'done' else '—'}"
                                            for l in r.get("levels", []))]]
            if r.get("observed_metrics"):
                facts.append(["métriques du projet", ", ".join(f"{k} {_fmt(v)}" for k, v in r["observed_metrics"].items())])
            waived = sorted({f["code"] for f in r.get("findings", []) if "levé par" in f.get("message", "")})
            if waived:
                facts.append(["bloquants levés par un humain", ", ".join(waived)])
            codes = sorted({f["code"] for f in r.get("findings", []) if f.get("severity") == "BLOCKER"})
            if codes:
                facts.append(["bloquants", ", ".join(codes[:6])])
    elif node == "repair":
        v2, rr, jv = _read(out.get("validator_v2")), _read(out.get("repair_report")), _read(out.get("judge_verdict"))
        if v2:
            facts += [["v2 : statut", f"{v2.get('status')} → {v2.get('next_action')}"],
                      ["v2 : problèmes", str(len(v2.get("issues", [])))]]
            extra["compare"] = compare_validators(ctx.get("v1") or {}, v2)
        if rr:
            facts += [["réparation", f"{rr.get('status')} · {len(rr.get('modifications', []))} modification(s)"]]
            if rr.get("requires_human_approval"):
                facts.append(["périmètre sensible", "oui : validation humaine"])
        if jv:
            facts.append(["juge", f"{jv.get('verdict')} ({jv.get('judge_model') or '?'})"
                                  + (f" · {jv.get('reason')[:120]}" if jv.get("reason") else "")])
    elif node == "agent1":
        spec, prof = _read(out.get("project_spec")), _read(out.get("data_profile"))
        if spec:
            facts += [["tâche · cible", f"{spec.get('task')} · {spec.get('target')}"],
                      ["métrique · seuil", f"{spec.get('primary_metric')} · {spec.get('min_threshold')}"],
                      ["features", str(len(spec.get("feature_columns", [])))]]
        if prof:
            facts.append(["lignes", str(prof.get("rows"))])
            if prof.get("class_balance"):
                facts.append(["classes", ", ".join(f"{k}: {v}" for k, v in prof["class_balance"].items())])
    elif node == "agent2":
        ex, meta = _read(out.get("experiment_result")), _read(out.get("mlflow_metadata"))
        if ex:
            m = ex.get("primary_metric", "")
            tried = ", ".join(f"{k} ({v} essais)" for k, v in (ex.get("trials_completed") or {}).items())
            facts += [["modèles essayés", tried or "—"], ["retenu", str(ex.get("best_model_name"))],
                      [f"{m} validation · test", f"{_fmt(ex.get('validation_score'))} · {_fmt(ex.get('test_score'))}"],
                      ["modèle naïf", _fmt(ex.get("baseline_test_score"))]]
            if ex.get("champion_test_score") is not None:
                facts.append(["champion (même jeu de test)", _fmt(ex["champion_test_score"])])
        if meta:
            g = meta.get("quality_gate") or {}
            facts.append(["quality gate", "réussi" if g.get("passed") else "raté : " + "; ".join(g.get("reasons", []))[:160]])
            facts.append(["registre", f"{meta.get('model_uri')} (alias {meta.get('alias')})"])
    elif node == "agent3":
        dep = _read(out.get("deployment_result"))
        if dep:
            checks = dep.get("checks", [])
            facts += [["modèle servi", f"{dep.get('model_uri')} ({dep.get('model_source')})"],
                      ["image", f"{dep.get('image')}:{dep.get('tag')}" + (" · Docker Hub" if dep.get("pushed") else "")],
                      ["contrôles", " · ".join(f"{c['name']} {'✓' if c['passed'] else '✗'}" for c in checks)],
                      ["API", dep.get("endpoint") or "non déployée (Docker absent)"]]
            facts += [[c["name"], c["details"]] for c in checks if c.get("details")][:2]
    elif node == "agent4":
        cfg = _read(out.get("monitoring_config"))
        if cfg:
            facts.append(["surveille", f"{cfg.get('model_uri', '?')} · fenêtres de {cfg.get('window_size', '?')}"])
        if out.get("retraining_request"):
            req = _read(out["retraining_request"])
            facts.append(["réentraînement demandé", f"{req.get('reason')} · {(req.get('diagnosis') or '')[:140]}"])
    elif node == "propose_manifest":
        man = _read(out.get("proposed_manifest"))
        if man:
            facts.append(["manifeste proposé", json.dumps(man, ensure_ascii=False)[:200]])
    return {"facts": facts, **extra}


def _station(node: str, entry: dict) -> dict:
    status = entry.get("status", "")
    if node == "validator" and status == "FAILED":
        kind = "blocked"
    elif status == "FAILED":
        kind = "failed"
    elif status == "WARNING" and not entry.get("can_proceed", True):
        kind = "stopped"
    else:
        kind = "passed"
    return {"id": node, "label": STATIONS.get(node, node), "status": kind, "agent_status": status,
            "duration_s": entry.get("duration_s"), "reasons": entry.get("reasons", [])[:3],
            "warnings": entry.get("warnings", [])[:3], "at": entry.get("started_at"),
            "all_reasons": entry.get("reasons", [])[:12], "all_warnings": entry.get("warnings", [])[:12]}


def track(run: Run, busy: bool) -> list[dict]:
    """Le plan de ligne : gares franchies (dans l'ordre réel, réparations et réentraînements compris),
    signal en attente, gare en cours, puis gares à venir."""
    s = run.state
    events = [(h.get("started_at") or "", "step", h) for h in s.get("history", [])]
    events += [(d.get("at") or "", "gate", d) for d in s.get("decisions", [])]
    events.sort(key=lambda e: e[0])
    out: list[dict] = []
    ctx: dict = {}                     # rapport du v1 vu juste avant (comparaison v1/v2 à la gare de réparation)
    last_run = {e["node"]: i for i, (_, kind, e) in enumerate(events) if kind == "step"}
    for i, (_, kind, e) in enumerate(events):
        if kind == "step":
            st = _station(e["node"], e)
            # le validateur et la réparation écrivent un dossier par passage ; les agents, un seul (réécrit)
            per_pass = e["node"] in ("validator", "repair") or last_run[e["node"]] == i
            try:
                st["details"] = (station_facts(e["node"], e, ctx) if per_pass else
                                 {"facts": [["fichiers", "remplacés par le passage suivant : voir ses raisons"]]})
            except Exception as exc:  # noqa: BLE001 - un fichier illisible ne casse pas le plan de ligne
                st["details"] = {"facts": [["détails indisponibles", type(exc).__name__]]}
            if st["id"] == "agent4" and out and out[-1]["id"] == "agent4":
                st["label"] += " · relecture"          # « relire la production » sans réentraînement
            out.append(st)
        else:
            node = GATE_NODE.get(e["gate"], e["gate"])
            out.append({"id": node, "label": STATIONS.get(node, node), "at": e.get("at"),
                        "status": "passed" if e["decision"] == "approve" else "refused",
                        "decided_by": e.get("by"), "comment": e.get("comment", "")})
    q = run.question
    if q:
        node = GATE_NODE.get(q["gate"], q["gate"])
        out.append({"id": node, "label": STATIONS.get(node, node), "status": "waiting", "gate": q["gate"]})
    elif busy and not s.get("final_status"):
        nxt = (s.get("supervision") or [{}])[-1].get("action", "")
        if nxt in STATIONS:
            out.append({"id": nxt, "label": STATIONS[nxt], "status": "running"})
    if not s.get("final_status"):
        reached = [i for i, n in enumerate(CANONICAL) if any(st["id"] == n for st in out)]
        for node in CANONICAL[(max(reached) + 1 if reached else 0):]:
            out.append({"id": node, "label": STATIONS[node], "status": "todo"})
    return out


def _model_served(state: dict) -> dict:
    dep = _read(state.get("artifacts", {}).get("deployment_result"))
    meta = _read(state.get("artifacts", {}).get("mlflow_metadata"))
    gate = meta.get("quality_gate") or {}
    return {"model_uri": dep.get("model_uri"), "endpoint": dep.get("endpoint"), "image": dep.get("image"),
            "pushed": dep.get("pushed"), "metric": gate.get("primary_metric"), "value": gate.get("value")}


def run_summary(run: Run, busy: bool) -> dict:
    s = run.state
    return {"run_id": run.run_id, "project": project_of(s), "project_dir": s.get("project_dir"),
            "status": "RUNNING" if busy and not s.get("final_status") and not run.question else run.status,
            "waiting_gate": run.question["gate"] if run.question else None,
            "retrain_count": s.get("retrain_count", 0), "repair_count": s.get("repair_count", 0),
            "supervisor": s.get("supervisor_mode"), "served": _model_served(s),
            "updated_at": datetime.fromtimestamp(run.run_dir.stat().st_mtime, timezone.utc).isoformat()
            if run.run_dir.exists() else None}


def gate_details(run: Run) -> dict | None:
    """Ce qu'il faut voir pour décider : la question, le fichier à relire et son contenu utile."""
    q = run.question
    if not q:
        return None
    d = {**q, "key": gate_key(run), "content": _read(q.get("file_to_review"))}
    if q["gate"] == "repair_review":
        outputs = run.state["results"]["repair"]["outputs"]
        diff = Path(outputs.get("repair_result", "")).with_name("repair.diff")
        d["diff"] = diff.read_text(encoding="utf-8") if diff.is_file() else ""
        d["judge"] = _read(outputs.get("judge_verdict"))
    if q["gate"] == "validator_override":
        d["validator_v1"] = _read(run.state.get("report_path"))
    return d


# ---------------------------------------------------------------------------
# Agent 4 en production : ce que voit sa gateway
# ---------------------------------------------------------------------------


def gateway_url() -> str:
    return os.environ.get("TERMINUS_GATEWAY_URL", "http://127.0.0.1:8100").rstrip("/")


def agent4_live(project: str) -> dict | None:
    """Fenêtres et décisions de l'agent de la gateway pour CE projet ; None si elle ne le surveille pas."""
    import httpx

    base = gateway_url()
    try:
        st = httpx.get(f"{base}/monitoring/status", timeout=1.5).json()
        if st.get("project") != project:
            return None
        decisions = httpx.get(f"{base}/monitoring/decisions", timeout=1.5).json()
        pending = []
        for d in decisions:
            if d.get("status") != "awaiting_human":
                continue
            full = httpx.get(f"{base}/monitoring/decisions/{d['id']}", timeout=1.5).json()
            sample = (full.get("samples") or [{}])[0]
            pending.append({**d, "root_cause": sample.get("root_cause"), "summary": sample.get("summary"),
                            "rationale": sample.get("rationale"),
                            "evidence": [{"fact": e.get("fact"), "value": e.get("value"), "claim": e.get("claim")}
                                         for e in sample.get("evidence", [])][:6],
                            "policy": full.get("policy")})
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    last = st.get("last_window") or {}
    return {"predictions": st.get("predictions"), "labels": st.get("labels"), "windows": st.get("windows"),
            "severity": last.get("severity"), "signals": last.get("signals", []),
            "decider": (st.get("agent") or {}).get("decider"), "pending": pending,
            "recent": [{k: d.get(k) for k in ("id", "action", "mode", "status", "window_index", "created_at")}
                       for d in decisions[-5:]]}


# ---------------------------------------------------------------------------
# Santé des services
# ---------------------------------------------------------------------------


def _http(url: str, timeout: float = 1.5) -> tuple[bool, str]:
    import httpx
    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code < 500, f"HTTP {r.status_code}"
    except httpx.HTTPError as exc:
        return False, type(exc).__name__


def _key_present(name: str, *env_files: Path) -> bool:
    return bool(os.environ.get(name)) or any(read_env_file(f).get(name) for f in env_files)


def health(runs_dir: str) -> list[dict]:
    out = [{"service": "base des runs", "where": describe_backend(runs_dir), "ok": True, "detail": "connectée"}]
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                           timeout=4, check=False)
        out.append({"service": "Docker", "where": "—", "ok": r.returncode == 0,
                    "detail": f"moteur {r.stdout.strip()}" if r.returncode == 0 else "moteur arrêté"})
    except (OSError, subprocess.TimeoutExpired):
        out.append({"service": "Docker", "where": "—", "ok": False, "detail": "docker introuvable"})
    mlflow_db = REPO / "runs" / "mlflow" / "mlflow.db"
    out.append({"service": "MLflow · registre", "where": os.environ.get("MLFLOW_TRACKING_URI") or "runs/mlflow",
                "ok": bool(os.environ.get("MLFLOW_TRACKING_URI")) or mlflow_db.exists(), "detail": ""})
    for service, url in [("API du modèle", "http://127.0.0.1:8000/health"),
                         ("Gateway Agent 4", os.environ.get("TERMINUS_GATEWAY_URL", "http://127.0.0.1:8100") + "/metrics"),
                         ("Prometheus", "http://127.0.0.1:9090/-/ready"),
                         ("Alertmanager", "http://127.0.0.1:9093/-/ready"),
                         ("Grafana", "http://127.0.0.1:3000/api/health"),
                         ("Décideur local MLX", "http://127.0.0.1:8080/v1/models")]:
        ok, detail = _http(url)
        out.append({"service": service, "where": url.split("/")[2], "ok": ok, "detail": detail,
                    "optional": service == "Décideur local MLX"})
    beat = Path(runs_dir) / ".slack_heartbeat"
    try:
        age = time.time() - float(beat.read_text())
    except (OSError, ValueError):
        age = None
    out.append({"service": "Service Slack", "where": "socket", "ok": age is not None and age < 30,
                "detail": "lancé" if age is not None and age < 30 else "arrêté (python -m orchestrator slack)"})
    agents = REPO / "agents"
    for name, files in [("GEMINI_API_KEY", (REPO / ".env",)),
                        ("GROQ_API_KEY", (agents / "agent1" / ".env", agents / "agent4" / ".env",
                                          agents / "validator" / ".env"))]:
        present = _key_present(name, *files)
        out.append({"service": f"clé {name.split('_')[0].title()}", "where": "—", "ok": present,
                    "detail": "présente" if present else "absente"})
    return out


# ---------------------------------------------------------------------------
# Registre MLflow
# ---------------------------------------------------------------------------


def _mlflow():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI")
                            or f"sqlite:///{REPO / 'runs' / 'mlflow' / 'mlflow.db'}")
    return MlflowClient()


def registry() -> list[dict]:
    c = _mlflow()
    out = []
    for m in c.search_registered_models():
        aliases = dict(getattr(m, "aliases", {}) or {})
        out.append({"name": m.name, "champion": aliases.get("champion"),
                    "latest": max((int(v.version) for v in m.latest_versions or []), default=None)})
    return out


def model_history(name: str) -> list[dict]:
    c = _mlflow()
    # search_model_versions ne remplit pas les alias : on les lit sur le modèle enregistré.
    by_version: dict[int, list[str]] = {}
    for alias, version in (getattr(c.get_registered_model(name), "aliases", {}) or {}).items():
        by_version.setdefault(int(version), []).append(alias)
    out = []
    for v in sorted(c.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version), reverse=True):
        run = c.get_run(v.run_id) if v.run_id else None
        metrics = dict(run.data.metrics) if run else {}
        params = dict(run.data.params) if run else {}
        out.append({"version": int(v.version), "aliases": by_version.get(int(v.version)) or list(getattr(v, "aliases", []) or []),
                    "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000, timezone.utc).isoformat(),
                    "metrics": {k: m for k, m in metrics.items() if k.startswith(("test_", "baseline_test_"))},
                    "model": params.get("best_model") or params.get("model"), "run_id": v.run_id})
    return out


# ---------------------------------------------------------------------------
# L'application
# ---------------------------------------------------------------------------


def create_app(runs_dir: str = "runs", sync: bool = False) -> FastAPI:
    """sync=True (tests) : les commandes s'exécutent avant la réponse, sans thread."""
    app = FastAPI(title="Terminus", description="API de la plateforme MLOps multi-agents")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    saver = get_checkpointer(runs_dir)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="terminus")   # agents lourds : un à la fois
    jobs: dict[str, dict] = {}
    names: dict[str, str] = {}          # projet d'une ligne lancée mais pas encore enregistrée
    lock = threading.Lock()

    def load(run_id: str) -> Run:
        try:
            return Run.load(run_id, runs_dir, saver)
        except RunNotFound:
            raise HTTPException(404, f"ligne {run_id} introuvable") from None

    def submit(run_id: str, label: str, work) -> None:
        def job():
            with lock:
                jobs[run_id] = {"job": label, "state": "en cours", "error": None}
            try:
                with run_lock(Path(runs_dir) / run_id):
                    work()
                jobs[run_id]["state"] = "terminé"
            except RunBusy as exc:
                jobs[run_id].update(state="refusé", error=str(exc))
            except Exception as exc:  # noqa: BLE001 - l'erreur est montrée dans l'interface
                jobs[run_id].update(state="erreur", error=f"{type(exc).__name__}: {exc}"[:500])
        if sync:
            job()
        else:
            pool.submit(job)

    @app.get("/api/health")
    def get_health():
        return health(runs_dir)

    @app.get("/api/projects")
    def projects():
        found = []
        for root in PROJECT_ROOTS:
            if root.is_dir():
                found += [{"name": d.name, "path": str(d.relative_to(REPO)), "manifest": (d / "mlops.yaml").exists()}
                          for d in sorted(root.iterdir()) if d.is_dir() and (d / "data.csv").exists()]
        return found

    @app.get("/api/runs")
    def runs():
        root = Path(runs_dir)
        out = []
        if root.exists():
            for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    run = Run.load(d.name, runs_dir, saver)
                except RunNotFound:
                    if d.name in jobs:          # tout juste lancée : pas encore d'étape enregistrée
                        out.append({"run_id": d.name, "status": "STARTING", **jobs[d.name]})
                    continue
                out.append(run_summary(run, is_busy(d)))
        return out

    @app.post("/api/runs", status_code=202)
    def start(req: StartRequest):
        project = (REPO / req.project_dir).resolve() if not Path(req.project_dir).is_absolute() else Path(req.project_dir)
        if not project.is_dir():
            raise HTTPException(400, f"{req.project_dir} n'est pas un dossier")
        run_id = uuid.uuid4().hex[:12]
        (Path(runs_dir) / run_id).mkdir(parents=True, exist_ok=True)
        names[run_id] = project.name
        submit(run_id, "lancement", lambda: Run.start(str(project), runs_dir, supervisor=req.supervisor,
                                                      run_id=run_id))
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        run_dir = (Path(runs_dir) / run_id).resolve()   # les chemins d'artefacts sont absolus
        try:
            run = Run.load(run_id, runs_dir, saver)
        except RunNotFound:
            if run_id in jobs or run_dir.exists():   # la ligne se forme : même forme qu'une ligne lancée
                track_ = [{"id": n, "label": STATIONS[n], "status": "running" if i == 0 else "todo"}
                          for i, n in enumerate(CANONICAL)]
                return {"run_id": run_id, "project": names.get(run_id), "status": "STARTING",
                        "job": jobs.get(run_id), "track": track_, "gate": None, "supervision": [],
                        "decisions": [], "results": {}, "artifacts": {}}
            raise HTTPException(404, f"ligne {run_id} introuvable") from None
        busy = is_busy(run_dir)
        s = run.state
        summary = run_summary(run, busy)
        live = agent4_live(summary["project"]) if (summary.get("served") or {}).get("endpoint") else None
        return {**summary, "track": track(run, busy), "gate": gate_details(run), "monitoring": live,
                "supervision": [{k: d.get(k) for k in ("after", "action", "explanation", "decider", "fallback", "at")}
                                for d in s.get("supervision", [])],
                "decisions": s.get("decisions", []), "results": s.get("results", {}),
                "artifacts": {k: str(Path(v).relative_to(run_dir)) if Path(v).is_relative_to(run_dir) else v
                              for k, v in s.get("artifacts", {}).items()},
                "job": jobs.get(run_id)}

    @app.get("/api/runs/{run_id}/file", response_class=PlainTextResponse)
    def run_file(run_id: str, path: str):
        run_dir = (Path(runs_dir) / run_id).resolve()
        target = (run_dir / path).resolve()
        if not target.is_relative_to(run_dir) or not target.is_file():
            raise HTTPException(404, "fichier introuvable dans cette ligne")
        if target.stat().st_size > 2_000_000:
            raise HTTPException(413, "fichier trop gros pour l'interface")
        return target.read_text(encoding="utf-8", errors="replace")

    @app.post("/api/runs/{run_id}/decision", status_code=202)
    def decide(run_id: str, req: DecisionRequest):
        run = load(run_id)
        if req.decision not in ("approve", "reject"):
            raise HTTPException(400, "decision : approve ou reject")
        if not run.question:
            raise HTTPException(409, "cette ligne n'attend aucune décision")
        if req.key and req.key != gate_key(run):
            raise HTTPException(409, "cette validation a déjà été traitée")
        if is_busy(run.run_dir):
            raise HTTPException(409, "la ligne est en cours d'exécution")
        if run.question["gate"] == "champion_replacement" and req.decision == "approve" and req.publish is not None:
            (run.run_dir / "publish.json").write_text(json.dumps({"push": req.publish, "by": req.by}),
                                                      encoding="utf-8")
        answer = {"decision": req.decision, "by": f"{req.by} (Terminus)", "comment": req.comment}
        submit(run_id, "décision", lambda: Run.load(run_id, runs_dir, saver).answer(answer))
        return {"run_id": run_id, "accepted": True}

    @app.post("/api/runs/{run_id}/monitor", status_code=202)
    def monitor(run_id: str):
        run = load(run_id)
        if run.status not in ("DONE", "CHAMPION_KEPT", "FAILED"):
            raise HTTPException(409, f"ligne {run.status} : le monitoring se relance sur une ligne déployée")
        submit(run_id, "monitoring", lambda: Run.load(run_id, runs_dir, saver).monitor())
        return {"run_id": run_id, "accepted": True}

    @app.post("/api/runs/{run_id}/agent4/{decision_id}", status_code=202)
    def agent4_answer(run_id: str, decision_id: str, req: Agent4Answer):
        """Répondre à une décision de l'agent de la gateway (comme la console ou Slack). Un réentraînement
        approuvé est aussitôt remis à la ligne : son signal « réentraînement » s'affiche ensuite ici."""
        import httpx

        if req.decision not in ("approve", "reject"):
            raise HTTPException(400, "decision : approve ou reject")
        run = load(run_id)
        try:
            r = httpx.post(f"{gateway_url()}/monitoring/decisions/{decision_id}/{req.decision}", timeout=120,
                           json={"by": f"{req.by} (Terminus)", "comment": req.comment})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"gateway injoignable ({type(exc).__name__})") from None
        if r.status_code >= 400:
            raise HTTPException(409, r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "")
                                else r.text)
        body = r.json()
        handed = req.decision == "approve" and (body.get("outputs") or {}).get("retraining_request")
        if handed and not is_busy(run.run_dir) and run.status in ("DONE", "CHAMPION_KEPT", "FAILED"):
            submit(run_id, "monitoring", lambda: Run.load(run_id, runs_dir, saver).monitor())
        return {"run_id": run_id, "decision": body.get("id"), "status": body.get("status"),
                "retraining": bool(handed)}

    @app.get("/api/decisions")
    def decisions():
        pending, done = [], []
        watched: set[str] = set()      # une décision de l'Agent 4 s'affiche sur la dernière ligne déployée du projet
        for r in runs():
            if r.get("status") == "STARTING":
                continue
            run = Run.load(r["run_id"], runs_dir, saver)
            g = gate_details(run)
            if g:
                pending.append({"run_id": run.run_id, "project": r["project"], "gate": g["gate"], "key": g["key"],
                                "question": g["question"]})
            if (r.get("served") or {}).get("endpoint") and r["project"] not in watched:
                watched.add(r["project"])
                for d in (agent4_live(r["project"]) or {}).get("pending", []):
                    pending.append({"run_id": run.run_id, "project": r["project"], "gate": "agent4_action",
                                    "key": d["id"], "question": f"L'Agent 4 propose : {d.get('action')}"})
            done += [{"run_id": run.run_id, "project": r["project"], **d} for d in run.state.get("decisions", [])]
        done.sort(key=lambda d: d.get("at") or "", reverse=True)
        return {"pending": pending, "recent": done[:20]}

    @app.get("/api/registry")
    def get_registry():
        return registry()

    @app.get("/api/registry/{name}")
    def get_model(name: str):
        return model_history(name)

    @app.get("/api/runs/{run_id}/example")
    def example(run_id: str):
        """Une ligne du jeu de données du projet, sans la cible : pré-remplit l'essai du modèle."""
        import pandas as pd
        spec = _read(load(run_id).state.get("artifacts", {}).get("project_spec"))
        project = Path(load(run_id).state.get("repaired_dir") or load(run_id).state["project_dir"])
        data = project / spec.get("data_path", "data.csv")
        if not data.is_file():
            raise HTTPException(404, "données du projet introuvables")
        row = pd.read_csv(data, nrows=1).drop(columns=[spec.get("target")], errors="ignore").iloc[0]
        return json.loads(row.to_json())

    @app.post("/api/predict")
    def predict(payload: dict):
        import httpx
        url = os.environ.get("TERMINUS_GATEWAY_URL", "http://127.0.0.1:8100") + "/predict"
        try:
            r = httpx.post(url, json=payload, timeout=10)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"gateway injoignable ({type(exc).__name__}) : lancer la gateway de l'Agent 4") from None
        return {"status": r.status_code, "body": r.json() if r.headers.get("content-type", "").startswith("application/json")
                else r.text}

    return app


def serve(runs_dir: str = "runs", port: int = 8090) -> int:
    import uvicorn
    print(f"Terminus API : http://127.0.0.1:{port}/api  ·  documentation : http://127.0.0.1:{port}/docs")
    uvicorn.run(create_app(runs_dir), host="127.0.0.1", port=port, log_level="warning")
    return 0
