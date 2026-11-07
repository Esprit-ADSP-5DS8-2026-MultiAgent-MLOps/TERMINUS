"""
Point d'entrée de l'Agent 4 pour l'orchestrateur (contrat : common/schemas.py).

    REAL["agent4"] = "monitoring_agent.pipeline:run"     # orchestrator/registry.py

À chaque appel :
1. mise en place (si l'Agent 3 a déployé une API : deployment_result.endpoint)
   référence notée par l'API, règles Prometheus + dashboard Grafana générés et validés ;
2. demande de la gateway : si l'agent de la gateway a déjà demandé un réentraînement
   (runs/<projet>/actions/retraining_request.json), elle est remise à l'orchestrateur, une seule fois ;
3. sinon, décision (si la gateway a déjà observé de la production pour ce projet)
   l'agent enquête sur la dernière fenêtre et décide ; une demande de réentraînement produit
   retraining_request.json, que l'orchestrateur soumet à sa propre validation humaine puis à l'Agent 2.
   Si la gateway tourne pour ce projet, c'est SON agent qui décide (MONITORING_GATEWAY_URL) : une seule
   instance, qui garde la mémoire d'incidents (Qdrant local ne s'ouvre que dans un processus à la fois).

L'agent n'échoue jamais pour une raison de monitoring : au pire il renvoie WARNING avec l'explication.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .agent.actions import _common_schemas
from .agent.graph import MonitoringAgent
from .agent.llm import decider_and_judge
from .provisioning.grafana import write_dashboard
from .provisioning.prometheus import check_rules, write_rules
from .runtime.store import Store
from .runtime.supervisor import Supervisor
from .runtime.target import MonitoringTarget, build_target, runs_dir

INFRA = Path(__file__).resolve().parent.parent / "infra"


def _reference_data(inp, dep) -> Path | None:
    """Les données d'entraînement du modèle déployé (Agent 2, mlflow_metadata.params.training_data) : c'est ce
    que le modèle a vu, donc la référence du monitoring. None si elles ne correspondent pas au modèle déployé
    (candidat refusé...) ou absentes : on garde alors la référence existante, sinon les données du projet."""
    path = inp.inputs.get("mlflow_metadata")
    if not path or not Path(path).is_file():
        return None
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    data = Path((meta.get("params") or {}).get("training_data") or "")
    return data if meta.get("model_uri") == dep.model_uri and data.is_file() else None


def _new_period(previous: MonitoringTarget | None, target: MonitoringTarget, model_uri: str, warnings: list) -> None:
    """Nouveau modèle servi : nouvelle période de monitoring. L'historique du modèle précédent est archivé
    (fenêtres, décisions, alertes) au lieu d'être mélangé avec celui du nouveau modèle."""
    if previous is None or previous.model_uri in ("unknown", model_uri) or not target.store_path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%S")
    tag = previous.model_uri.removeprefix("models:/").replace("/", "_v")
    archive = target.store_path.with_name(f"monitoring_{tag}_{stamp}.db")
    for suffix in ("", "-wal", "-shm"):                     # SQLite en mode WAL
        f = Path(f"{target.store_path}{suffix}")
        if f.exists():
            f.rename(Path(f"{archive}{suffix}"))
    warnings.append(f"nouveau modèle {model_uri} : nouvelle période de monitoring, historique de "
                    f"{previous.model_uri} archivé ({archive.name}) ; relancer la gateway")


def _setup(inp, spec, dep, outputs: dict, warnings: list) -> MonitoringTarget | None:
    if dep is None or not dep.endpoint:
        warnings.append("pas d'API déployée (deployment_result.endpoint vide) : monitoring en ligne non configuré")
        try:
            return MonitoringTarget.load(spec.project_name)
        except FileNotFoundError:
            return None
    try:
        previous = MonitoringTarget.load(spec.project_name)
    except FileNotFoundError:
        previous = None
    reference = _reference_data(inp, dep)
    if reference is None and previous and previous.model_uri == dep.model_uri and previous.reference_path.exists():
        # même modèle, données d'entraînement inconnues : on garde sa référence (ne pas revenir aux données du projet)
        target, info = previous, {"failed": 0, "scored": previous.reference_rows}
        target.upstream_url = dep.endpoint.rstrip("/")
    else:
        try:
            target, info = build_target(inp.project_dir, dep.endpoint, name=spec.project_name,
                                        reference_data=reference)
        except Exception as exc:  # noqa: BLE001 - API injoignable, données absentes... : on explique, on ne plante pas
            warnings.append(f"mise en place du monitoring impossible : {type(exc).__name__}: {exc}")
            return None
    if reference is not None:
        warnings.append(f"référence : données d'entraînement de {dep.model_uri} ({info['scored']} lignes)")
    _new_period(previous, target, dep.model_uri, warnings)
    target.model_uri = dep.model_uri
    target.save()
    rules = write_rules(target, INFRA / "prometheus" / "rules")
    ok, msg = check_rules(rules)
    if ok is False:
        warnings.append(f"règles Prometheus invalides : {msg[:300]}")
    dash = write_dashboard(target, INFRA / "grafana" / "dashboards")
    wd = Path(inp.workdir)
    for src in (target.dir / "monitoring.json", rules, dash):
        shutil.copy(src, wd / src.name)
    outputs.update(monitoring_config=str(wd / "monitoring.json"), alert_rules=str(wd / rules.name),
                   dashboard=str(wd / dash.name))
    if info["failed"]:
        warnings.append(f"{info['failed']} lignes de référence non notées par l'API")
    return target


def _gateway_request(target: MonitoringTarget, workdir: Path, outputs: dict, warnings: list) -> bool:
    """Demande de réentraînement décidée par la gateway, pas encore remise à l'orchestrateur.

    Copiée dans le workdir (avec ses nouvelles données), puis marquée remise : elle n'est donnée qu'une fois,
    que l'humain l'approuve ou non à la gate « retraining ».
    """
    src = target.dir / "actions" / "retraining_request.json"
    if not src.is_file():
        return False
    req = json.loads(src.read_text(encoding="utf-8"))
    data = Path(req.get("new_data_path") or "")
    if data.is_file():
        shutil.copy(data, workdir / data.name)
        req["new_data_path"] = str(workdir / data.name)
        outputs["new_data"] = str(workdir / data.name)
    dest = workdir / "retraining_request.json"
    dest.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    src.rename(src.with_name(f"retraining_request_{time.strftime('%Y%m%dT%H%M%S')}.handed.json"))
    outputs["retraining_request"] = str(dest)
    warnings.append(f"demande de réentraînement de la gateway remise à l'orchestrateur ({req.get('reason')}) : "
                    f"{(req.get('diagnosis') or '')[:200]}")
    return True


def gateway_url() -> str:
    return os.environ.get("MONITORING_GATEWAY_URL", "http://127.0.0.1:8100").rstrip("/")


def _live_gateway(project: str) -> dict | None:
    """Statut de la gateway si elle tourne POUR CE projet (sinon None : l'agent décide ici)."""
    import httpx

    try:
        st = httpx.get(f"{gateway_url()}/monitoring/status", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        return None
    return st if st.get("project") == project and st.get("agent") else None


def _decide_in_gateway(target: MonitoringTarget, status: dict, workdir: Path, outputs: dict, warnings: list) -> bool:
    """La gateway tourne : c'est SON agent qui décide (une seule instance, propriétaire de la mémoire Qdrant
    locale, qu'un second processus ne peut pas ouvrir). Renvoie True si une décision a été prise ou remise."""
    import httpx

    base = gateway_url()
    pending = [d for d in httpx.get(f"{base}/monitoring/decisions", timeout=5).json()
               if d.get("status") == "awaiting_human"]
    if pending:
        d = pending[0]
        warnings.append(f"décision {d['id']} de l'agent de la gateway en attente d'un humain ({d.get('action')}) : "
                        "à valider dans Terminus, la console ou Slack")
        return True
    if status["agent"].get("busy"):
        warnings.append("l'agent de la gateway est déjà en train de décider : relancer dans un instant")
        return True
    last = status.get("last_window") or {}
    if not last.get("trigger_agent"):
        idx = (last.get("window") or {}).get("index")
        warnings.append(f"dernière fenêtre ({idx}) sans signal : rien à décider (vu par la gateway)")
        return False
    run = httpx.post(f"{base}/monitoring/decide", timeout=600).json()
    p = run.get("policy") or {}
    outputs["decision"] = str(workdir / f"decision_{run['id']}.json")
    Path(outputs["decision"]).write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    warnings.append(f"agent (gateway) : {p.get('proposed_action')} -> {p.get('final_action')} "
                    f"({p.get('mode')}, {run.get('status')})")
    if p.get("final_action") == "request_retraining" and run.get("status") == "awaiting_human":
        # comme en local : la gate « retraining » de l'orchestrateur est l'humain qui valide
        httpx.post(f"{base}/monitoring/decisions/{run['id']}/approve", timeout=600,
                   json={"by": "orchestrateur (gate humaine en aval)", "comment": ""})
        _gateway_request(target, workdir, outputs, warnings)
    return True


def _keep_files(src: dict, outputs: dict, warnings: list) -> None:
    """Le contrat attend des chemins dans outputs : le reste (canal de notification...) va dans warnings."""
    for key, value in src.items():
        if value and Path(value).exists():
            outputs[key] = value
        else:
            warnings.append(f"{key} : {value}")


def run(inp):
    s = _common_schemas()
    started = time.monotonic()
    outputs: dict[str, str] = {}
    warnings: list[str] = []
    spec = s.ProjectSpec.model_validate_json(Path(inp.inputs["project_spec"]).read_text(encoding="utf-8"))
    dep = None
    if "deployment_result" in inp.inputs:
        dep = s.DeploymentResult.model_validate_json(Path(inp.inputs["deployment_result"]).read_text(encoding="utf-8"))

    target = _setup(inp, spec, dep, outputs, warnings)
    status = s.AgentStatus.OK

    live = _live_gateway(target.project) if target else None
    if target and _gateway_request(target, Path(inp.workdir), outputs, warnings):
        status = s.AgentStatus.WARNING
    elif target and live is not None:
        try:
            if _decide_in_gateway(target, live, Path(inp.workdir), outputs, warnings):
                status = s.AgentStatus.WARNING
        except Exception as exc:  # noqa: BLE001 - gateway injoignable en cours de route
            warnings.append(f"gateway injoignable pendant la décision ({type(exc).__name__}) : relancer")
    elif target and target.store_path.exists():
        store = Store(target.store_path)
        try:
            decider, judge = decider_and_judge()
            from .memory.store import default_memory
            agent = MonitoringAgent(decider, judge, out_dir=Path(inp.workdir), memory=default_memory())
            sup = Supervisor(target, store, agent=agent)
            case = sup.build_case()
            if case.current.trigger_agent:
                result = agent.decide(case)
                outputs["decision"] = str(Path(inp.workdir) / f"decision_{result.id}.json")
                Path(outputs["decision"]).write_text(result.model_dump_json(indent=2), encoding="utf-8")
                _keep_files(result.outputs, outputs, warnings)
                p = result.policy
                if p:
                    warnings.append(f"agent : {p.proposed_action} -> {p.final_action} ({p.mode}, {result.status})")
                    if p.final_action == "request_retraining" and "retraining_request" not in outputs:
                        # En attente d'un humain : la gate « retraining » de l'orchestrateur est cet humain.
                        agent.resume(result.id, "approve", by="orchestrateur (gate humaine en aval)")
                        _keep_files(agent.result(result.id).outputs, outputs, warnings)
                status = s.AgentStatus.WARNING
            else:
                warnings.append(f"dernière fenêtre ({case.current.window.index}) sans signal : rien à décider")
            sup.close()
        except RuntimeError as exc:
            warnings.append(str(exc))
        finally:
            store.close()
    elif target:
        warnings.append("monitoring prêt ; aucune prédiction observée pour l'instant "
                        f"(lancer : python -m monitoring_agent gateway {target.project})")

    if warnings and status == s.AgentStatus.OK:
        status = s.AgentStatus.WARNING
    return s.AgentResult(agent="agent4", status=status, can_proceed=True, outputs=outputs, warnings=warnings,
                         duration_s=round(time.monotonic() - started, 3))


__all__ = ["run", "runs_dir"]
