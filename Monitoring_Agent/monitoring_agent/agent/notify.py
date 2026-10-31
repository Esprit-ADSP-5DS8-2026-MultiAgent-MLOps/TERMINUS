"""
Notifications : prévenir une vraie personne quand l'agent agit ou attend une validation.

Trois sorties, de la plus riche à la plus simple :
- l'app Slack (SLACK_BOT_TOKEN + SLACK_CHANNEL) : message avec boutons Approuver / Rejeter quand l'agent
  attend un humain ; la suite (action exécutée, rejet) arrive dans le fil du même message ;
- un webhook entrant (Slack, Teams, Discord, Mattermost) si MONITORING_NOTIFY_WEBHOOK est défini ;
- toujours une ligne dans notifications.jsonl (dossier du projet), lisible par l'orchestrateur.

Les clics sur les boutons sont reçus par runtime/slack.py (Socket Mode). Ici on ne fait qu'envoyer.
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
APPROVE, REJECT, RATE_UP = "agent4_approve", "agent4_reject", "agent4_decision_up"


def console_link(run_id: str) -> str:
    base = _settings().get("MONITORING_PUBLIC_URL", "http://127.0.0.1:8100").rstrip("/")
    return f"{base}/console/#{run_id}"


def build_message(run, project: str, event: str) -> dict:
    """`event` : "awaiting_human" (il faut valider), "executed" (l'action est faite) ou "rejected"."""
    d, p = run.diagnosis, run.policy
    action = p.final_action if p else (d.action if d else "?")
    who = (run.human or {}).get("by")
    if event == "awaiting_human":
        title = f"[{project}] Validation demandée : {action}"
        why = "; ".join(p.reasons) if p and p.reasons else "la policy demande un humain"
        lines = [f"L'agent propose « {action} » mais ne peut pas le faire seul : {why}."]
    elif event == "rejected":
        title = f"[{project}] Action rejetée : {action}"
        lines = [f"« {action} » a été rejeté{f' par {who}' if who else ''}. Rien n'a été exécuté."]
    else:
        title = f"[{project}] {NOTIFY_ACTIONS.get(action, action)}"
        mode = f"validée par {who}" if who else f"mode {p.mode if p else '?'}"
        lines = [f"Action exécutée : {action} ({mode})."]
    if d:
        lines.append(f"Cause : {d.root_cause} · confiance {d.confidence:.2f} · fenêtre {run.window_index}")
        lines.append(d.summary)
    link = console_link(run.id)
    lines.append(f"Détail et décision : {link}")
    return {"title": title, "text": "\n".join(lines), "link": link, "decision": run.id, "project": project,
            "action": action, "event": event}


# ---------------------------------------------------------------------------
# Webhook entrant
# ---------------------------------------------------------------------------


def _payload(url: str, msg: dict) -> dict:
    """Chaque outil attend un champ différent : Discord `content`, Slack / Mattermost / Teams `text`."""
    body = f"*{msg['title']}*\n{msg['text']}"
    if "discord.com" in url or "discordapp.com" in url:
        return {"content": body.replace("*", "**")}
    return {"text": body}


# ---------------------------------------------------------------------------
# App Slack (boutons, fils)
# ---------------------------------------------------------------------------


def slack_settings() -> dict[str, str]:
    s = _settings()
    return {k: s.get(k, "").strip() for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL",
                                              "SLACK_APPROVERS")}


def _slack_client(token: str):
    # importé ici : sans Slack configuré, le paquet n'est pas nécessaire
    from slack_sdk import WebClient
    return WebClient(token=token)


def slack_blocks(msg: dict, buttons: bool) -> list[dict]:
    body = msg["text"].rsplit("\nDétail et décision", 1)[0]
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*{msg['title']}*\n{body}"}}]
    elements: list[dict] = []
    if buttons:
        elements += [
            {"type": "button", "action_id": APPROVE, "value": msg["decision"], "style": "primary",
             "text": {"type": "plain_text", "text": "Approuver"}},
            {"type": "button", "action_id": REJECT, "value": msg["decision"], "style": "danger",
             "text": {"type": "plain_text", "text": "Rejeter"},
             "confirm": {"title": {"type": "plain_text", "text": "Rejeter l'action ?"},
                         "text": {"type": "plain_text", "text": "Rien ne sera exécuté."},
                         "confirm": {"type": "plain_text", "text": "Rejeter"},
                         "deny": {"type": "plain_text", "text": "Annuler"}}},
        ]
    elif msg["event"] == "executed":
        elements.append({"type": "button", "action_id": RATE_UP, "value": msg["decision"],
                         "text": {"type": "plain_text", "text": "Bonne décision"}})
    elements.append({"type": "button", "action_id": "agent4_open", "url": msg["link"],
                     "text": {"type": "plain_text", "text": "Ouvrir dans la console"}})
    blocks.append({"type": "actions", "elements": elements})
    return blocks


def _thread_of(out_dir: Path, decision: str) -> dict | None:
    """Le message Slack déjà envoyé pour cette décision (celui qui attendait une validation)."""
    path = out_dir / "notifications.jsonl"
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("decision") == decision and rec.get("slack_ts") and not rec.get("slack_thread"):
            return rec
    return None


def _send_slack(msg: dict, conf: dict, out_dir: Path) -> None:
    client = _slack_client(conf["SLACK_BOT_TOKEN"])
    channel = conf["SLACK_CHANNEL"]
    parent = _thread_of(out_dir, msg["decision"]) if msg["event"] != "awaiting_human" else None
    blocks = slack_blocks(msg, buttons=msg["event"] == "awaiting_human")
    if parent:
        # La suite d'une validation : le message d'origine perd ses boutons, la suite va dans son fil.
        status = "Approuvé et exécuté" if msg["event"] == "executed" else "Rejeté"
        client.chat_update(channel=parent["slack_channel"], ts=parent["slack_ts"], text=parent["title"],
                           blocks=[{"type": "section", "text": {"type": "mrkdwn",
                                                                "text": f"*{parent['title']}*\n_{status}._"}}])
        r = client.chat_postMessage(channel=parent["slack_channel"], thread_ts=parent["slack_ts"],
                                    text=msg["title"], blocks=blocks)
        msg.update(slack_channel=parent["slack_channel"], slack_ts=r["ts"], slack_thread=parent["slack_ts"])
    else:
        r = client.chat_postMessage(channel=channel, text=msg["title"], blocks=blocks)
        msg.update(slack_channel=r["channel"], slack_ts=r["ts"])


# ---------------------------------------------------------------------------


def send(run, project: str, event: str, out_dir: Path) -> str:
    """Envoie la notification ; renvoie le canal utilisé ("slack", "webhook" ou "fichier")."""
    msg = build_message(run, project, event)
    conf = slack_settings()
    channel = "fichier"
    if conf["SLACK_BOT_TOKEN"] and conf["SLACK_CHANNEL"]:
        try:
            _send_slack(msg, conf, out_dir)
            channel = "slack"
        except Exception as exc:  # noqa: BLE001 - paquet absent, jeton invalide, bot pas invité dans le canal
            log.warning("notification Slack impossible : %s", exc)
            msg["error"] = f"slack : {type(exc).__name__}: {exc}"[:200]
    url = _settings().get("MONITORING_NOTIFY_WEBHOOK", "").strip()
    if url and channel == "fichier":
        try:
            r = httpx.post(url, json=_payload(url, msg), timeout=10)
            r.raise_for_status()
            channel = "webhook"
        except httpx.HTTPError as exc:
            log.warning("notification webhook impossible : %s", exc)
            msg["error"] = f"{type(exc).__name__}: {exc}"[:200]
    if msg.get("error") and channel == "fichier":
        channel = "fichier (envoi en échec)"
    msg.update(channel=channel, sent_at=datetime.now(UTC).isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "notifications.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return channel
