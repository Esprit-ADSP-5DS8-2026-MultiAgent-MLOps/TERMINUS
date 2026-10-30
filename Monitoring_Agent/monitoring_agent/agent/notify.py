"""
Notifications : prévenir une vraie personne quand l'agent agit ou attend une validation.

Deux sorties :
- un webhook (Slack, Microsoft Teams, Discord, Mattermost...) si MONITORING_NOTIFY_WEBHOOK est défini ;
- toujours une ligne dans notifications.jsonl (dossier du projet), lisible par l'orchestrateur ou un autre outil.

Une notification qui échoue ne bloque jamais la décision : on l'écrit dans le fichier avec l'erreur.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .llm import _settings

log = logging.getLogger("monitoring_agent.notify")

# Actions qui méritent de déranger quelqu'un. increase_monitoring et no_action restent silencieuses.
NOTIFY_ACTIONS = {
    "notify_data_team": "Problème de qualité des données",
    "open_incident": "Incident de monitoring",
    "request_retraining": "Réentraînement demandé",
    "rollback_model": "Retour au modèle précédent",
}


def console_link(run_id: str) -> str:
    base = _settings().get("MONITORING_PUBLIC_URL", "http://127.0.0.1:8100").rstrip("/")
    return f"{base}/console/#{run_id}"


def build_message(run, project: str, event: str) -> dict:
    """`event` : "executed" (l'action est faite) ou "awaiting_human" (il faut valider)."""
    d, p = run.diagnosis, run.policy
    action = p.final_action if p else (d.action if d else "?")
    if event == "awaiting_human":
        title = f"[{project}] Validation demandée : {action}"
        why = "; ".join(p.reasons) if p and p.reasons else "la policy demande un humain"
        lines = [f"L'agent propose « {action} » mais ne peut pas le faire seul : {why}."]
    else:
        title = f"[{project}] {NOTIFY_ACTIONS.get(action, action)}"
        lines = [f"Action exécutée : {action} (mode {p.mode if p else '?'})."]
    if d:
        lines.append(f"Cause : {d.root_cause} · confiance {d.confidence:.2f} · fenêtre {run.window_index}")
        lines.append(d.summary)
    link = console_link(run.id)
    lines.append(f"Détail et décision : {link}")
    return {"title": title, "text": "\n".join(lines), "link": link, "decision": run.id, "project": project,
            "action": action, "event": event}


def _payload(url: str, msg: dict) -> dict:
    """Chaque outil attend un champ différent : Discord `content`, Slack / Mattermost / Teams `text`."""
    body = f"*{msg['title']}*\n{msg['text']}"
    if "discord.com" in url or "discordapp.com" in url:
        return {"content": body.replace("*", "**")}
    return {"text": body}


def send(run, project: str, event: str, out_dir: Path) -> str:
    """Envoie la notification ; renvoie le canal utilisé ("webhook", "fichier" ou "fichier (webhook en échec)")."""
    msg = build_message(run, project, event)
    url = _settings().get("MONITORING_NOTIFY_WEBHOOK", "").strip()
    channel = "fichier"
    if url:
        try:
            r = httpx.post(url, json=_payload(url, msg), timeout=10)
            r.raise_for_status()
            channel = "webhook"
        except httpx.HTTPError as exc:
            log.warning("notification webhook impossible : %s", exc)
            msg["error"] = f"{type(exc).__name__}: {exc}"[:200]
            channel = "fichier (webhook en échec)"
    msg.update(channel=channel, sent_at=datetime.now(UTC).isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "notifications.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return channel
