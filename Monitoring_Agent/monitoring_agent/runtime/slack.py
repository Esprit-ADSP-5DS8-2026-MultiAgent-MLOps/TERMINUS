"""
L'app Slack de l'agent, en Socket Mode (connexion sortante vers Slack : ni port ouvert, ni URL publique).

Ce qu'on reçoit de Slack :
- un clic sur Approuver / Rejeter sous une décision qui attend un humain -> reprise du graphe, comme
  le bouton de la console (la suite arrive dans le fil du message, voir agent/notify.py) ;
- un clic sur « Bonne décision » -> avis gardé pour le fine-tuning ;
- /monitoring <question> ou une mention @agent4 -> réponse du chat « Demande à ton monitoring »,
  avec deux boutons Utile / Pas utile.

Seules les personnes listées dans SLACK_APPROVERS peuvent approuver ou rejeter (liste vide = tout le canal).
Le nom Slack de la personne est gardé dans la décision (« par Mohamed (Slack) »).

Configuration dans .env : SLACK_BOT_TOKEN (xoxb-...), SLACK_APP_TOKEN (xapp-...), SLACK_CHANNEL (C...),
SLACK_APPROVERS (U...,U...).
"""

from __future__ import annotations

import logging
import re
import threading

from ..agent.notify import APPROVE, RATE_UP, REJECT, slack_settings

log = logging.getLogger("monitoring_agent.slack")

ANSWER_UP, ANSWER_DOWN = "agent4_answer_up", "agent4_answer_down"
MENTION = re.compile(r"<@[A-Z0-9]+>\s*")


def to_mrkdwn(text: str) -> str:
    """Markdown du LLM -> format Slack (gras avec une seule étoile)."""
    return re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)


class SlackBridge:
    def __init__(self, target, store, supervisor, settings: dict[str, str] | None = None):
        self.target, self.store, self.supervisor = target, store, supervisor
        self.conf = settings or slack_settings()
        self.approvers = {u.strip() for u in self.conf.get("SLACK_APPROVERS", "").split(",") if u.strip()}
        self.handler = None

    @property
    def configured(self) -> bool:
        return bool(self.conf.get("SLACK_BOT_TOKEN") and self.conf.get("SLACK_APP_TOKEN"))

    # -- logique (testable sans Slack) ---------------------------------------------------

    def can_approve(self, user_id: str) -> bool:
        return not self.approvers or user_id in self.approvers

    def on_decision(self, user_id: str, user_name: str, run_id: str, answer: str) -> str:
        """Réponse à un clic Approuver / Rejeter. Renvoie le message à montrer à la personne."""
        if not self.can_approve(user_id):
            return "Tu n'es pas dans la liste des personnes autorisées à valider (SLACK_APPROVERS)."
        if self.supervisor is None:
            return "L'agent est désactivé sur cette gateway."
        try:
            run = self.supervisor.resume(run_id, answer, f"{user_name} (Slack)", "via Slack")
        except (KeyError, RuntimeError):
            return f"La décision {run_id} n'attend plus de réponse (déjà traitée, ou gateway redémarrée)."
        if answer == "approve":
            return f"Approuvé : {run.policy.final_action if run.policy else 'action'} exécuté ({run.status})."
        return "Rejeté : rien n'a été exécuté."

    def on_rating(self, user_name: str, kind: str, ref: str, rating: str) -> str:
        from .. import ratings
        try:
            res = ratings.rate(self.target.dir, self.store, self.supervisor, kind, ref, rating,
                               by=f"{user_name} (Slack)")
        except (KeyError, ValueError) as exc:
            return f"Avis non enregistré : {exc}"
        extra = " Il devient un exemple d'entraînement." if res["training_example"] else ""
        return f"Merci, avis enregistré.{extra}"

    def on_question(self, question: str) -> dict:
        from ..agent.assistant import ask
        from ..agent.mcp_server import server_for_target
        question = MENTION.sub("", question).strip()
        if not question:
            return {"answer": "Pose ta question après la commande, par exemple : /monitoring le modèle va bien ?"}
        try:
            case = self.supervisor.build_case(detailed=True)
        except RuntimeError:
            return {"answer": "Aucune fenêtre analysée pour l'instant : il faut d'abord du trafic sur /predict."}
        return ask(question, case, server_for_target(self.target, self.store), log_dir=self.target.dir)

    def answer_blocks(self, question: str, rec: dict, who: str = "") -> list[dict]:
        head = f"*{who + ' : ' if who else ''}{question}*\n" if question else ""
        blocks: list[dict] = [{"type": "section",
                               "text": {"type": "mrkdwn", "text": head + to_mrkdwn(rec["answer"])[:2900]}}]
        tools = ", ".join(t["tool"] for t in rec.get("tools", []))
        if rec.get("model"):
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text":
                           f"{rec['model']}{' · outils : ' + tools if tools else ''}"}]})
        if rec.get("id"):
            blocks.append({"type": "actions", "elements": [
                {"type": "button", "action_id": ANSWER_UP, "value": rec["id"],
                 "text": {"type": "plain_text", "text": "Utile"}},
                {"type": "button", "action_id": ANSWER_DOWN, "value": rec["id"],
                 "text": {"type": "plain_text", "text": "Pas utile"}}]})
        return blocks

    # -- branchement Slack -----------------------------------------------------------------

    def start(self) -> bool:
        """Ouvre la connexion Socket Mode dans un thread. Renvoie False si Slack n'est pas configuré."""
        if not self.configured:
            return False
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
        except ImportError:
            log.warning("Slack configuré mais slack-bolt n'est pas installé (pip install slack-bolt)")
            return False
        try:
            app = App(token=self.conf["SLACK_BOT_TOKEN"])     # vérifie le jeton auprès de Slack
        except Exception as exc:  # noqa: BLE001 - jeton refusé, réseau : la gateway doit démarrer quand même
            log.warning("Slack : jeton bot refusé (%s) ; seules les notifications par fichier restent", exc)
            return False

        def name_of(body: dict) -> tuple[str, str]:
            u = body.get("user", {})
            return u.get("id", ""), u.get("name") or u.get("username") or u.get("id", "?")

        @app.action(APPROVE)
        @app.action(REJECT)
        def decision_click(ack, body, respond, action):
            ack()
            user_id, user_name = name_of(body)
            answer = "approve" if action["action_id"] == APPROVE else "reject"
            respond(response_type="ephemeral", replace_original=False,
                    text=self.on_decision(user_id, user_name, action["value"], answer))

        @app.action(RATE_UP)
        def decision_up(ack, body, respond, action):
            ack()
            respond(response_type="ephemeral", replace_original=False,
                    text=self.on_rating(name_of(body)[1], "decision", action["value"], "up"))

        @app.action(ANSWER_UP)
        @app.action(ANSWER_DOWN)
        def answer_rating(ack, body, respond, action):
            ack()
            rating = "up" if action["action_id"] == ANSWER_UP else "down"
            respond(response_type="ephemeral", replace_original=False,
                    text=self.on_rating(name_of(body)[1], "answer", action["value"], rating))

        @app.action("agent4_open")
        def open_link(ack):
            ack()     # bouton-lien : Slack ouvre l'URL, il faut seulement accuser réception

        @app.command("/monitoring")
        def slash(ack, command, respond):
            ack(text="Je regarde le monitoring...")
            rec = self.on_question(command.get("text", ""))
            respond(response_type="in_channel", text=rec["answer"][:300],
                    blocks=self.answer_blocks(command.get("text", ""), rec, command.get("user_name", "")))

        @app.event("app_mention")
        def mention(event, say):
            rec = self.on_question(event.get("text", ""))
            say(text=rec["answer"][:300], blocks=self.answer_blocks("", rec),
                thread_ts=event.get("thread_ts") or event.get("ts"))

        self.handler = SocketModeHandler(app, self.conf["SLACK_APP_TOKEN"])
        threading.Thread(target=self._connect, daemon=True, name="slack").start()
        return True

    def _connect(self) -> None:
        try:
            self.handler.connect()
            log.info("Slack connecté (Socket Mode)")
        except Exception:
            log.exception("connexion Slack impossible (vérifie SLACK_APP_TOKEN et le Socket Mode)")

    def close(self) -> None:
        if self.handler is not None:
            try:
                self.handler.close()
            except Exception:  # fermeture best effort
                log.debug("fermeture Slack", exc_info=True)
