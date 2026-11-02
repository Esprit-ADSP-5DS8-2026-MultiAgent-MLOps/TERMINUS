"""
Les validations humaines de l'orchestrateur dans Slack (2e app Slack, distincte de celle de l'Agent 4).

    python -m orchestrator slack                       # le service : reste lancé pendant la démo
    python -m orchestrator run <projet> --slack        # le run s'arrête aux gates, la décision se prend dans Slack

Le service, en Socket Mode (connexion sortante : ni port ouvert, ni URL publique) :
- poste chaque gate en attente avec ses boutons, dans un fil par run :
  manifest_confirmation, champion_replacement (avec ou sans publication Docker Hub), retraining, failure_review ;
- reprend le run quand une personne autorisée clique (le pipeline continue dans ce processus) ;
- relance l'Agent 4 (`monitor`) quand sa gateway a demandé un réentraînement : la gate `retraining` arrive dans Slack ;
- annonce le début et la fin de chaque run.

Qui valide : ORCH_SLACK_APPROVERS_<GATE> (ex. ORCH_SLACK_APPROVERS_CHAMPION_REPLACEMENT=U01,U02), sinon
ORCH_SLACK_APPROVERS, sinon tout le canal. La publication Docker Hub a sa propre liste (…_DOCKER_PUSH).
Jamais de jeton ni d'identifiant Docker Hub dans Slack : le compte et le jeton restent dans agents/agent3/.env.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from common.envscope import read_env_file

from .graph import Run, RunNotFound

log = logging.getLogger("orchestrator.slack")

REPO = Path(__file__).resolve().parents[1]
APPROVE, PUBLISH, REJECT = "orch_approve", "orch_approve_publish", "orch_reject"
STATE_FILE = "slack.json"          # dans runs/<run_id>/ : ce qui a déjà été posté
PUBLISH_FILE = "publish.json"      # dans runs/<run_id>/ : choix de publication lu par l'Agent 3
FINAL = {"DONE", "BLOCKED", "FAILED", "REJECTED", "MANIFEST_PROPOSED", "CHAMPION_KEPT"}
TITLES = {
    "manifest_confirmation": "Manifeste proposé par l'Agent 1",
    "champion_replacement": "Nouveau champion à valider",
    "retraining": "Réentraînement demandé par l'Agent 4",
    "failure_review": "Étape en échec",
}
FINAL_TEXT = {
    "DONE": ":white_check_mark: terminé : modèle déployé et surveillé",
    "CHAMPION_KEPT": ":shield: champion gardé : le modèle en production reste en place",
    "BLOCKED": ":no_entry: bloqué par le validateur",
    "FAILED": ":x: échec",
    "REJECTED": ":hand: refusé par un humain",
    "MANIFEST_PROPOSED": ":memo: manifeste validé : ajouter le mlops.yaml au projet puis relancer",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _ids(value: str | None) -> set[str]:
    return {u.strip() for u in (value or "").split(",") if u.strip()}


class Settings:
    def __init__(self, env: dict[str, str] | None = None):
        e = os.environ if env is None else env
        self.bot_token = e.get("ORCH_SLACK_BOT_TOKEN", "")
        self.app_token = e.get("ORCH_SLACK_APP_TOKEN", "")
        self.channel = e.get("ORCH_SLACK_CHANNEL", "")
        self.default = _ids(e.get("ORCH_SLACK_APPROVERS"))
        self.per_gate = {k.removeprefix("ORCH_SLACK_APPROVERS_").lower(): _ids(v) for k, v in e.items()
                         if k.startswith("ORCH_SLACK_APPROVERS_") and _ids(v)}
        self.poll_s = float(e.get("ORCH_SLACK_POLL_S", "5"))

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.app_token and self.channel)

    def approvers(self, gate: str) -> set[str]:
        """Liste vide = tout le canal."""
        return self.per_gate.get(gate) or self.default

    def can(self, gate: str, user_id: str) -> bool:
        allowed = self.approvers(gate)
        return not allowed or user_id in allowed


def _agent3_env() -> dict[str, str]:
    return {**read_env_file(REPO / "agents" / "agent3" / ".env"),
            **{k: v for k, v in os.environ.items() if k.startswith(("AGENT3_", "DOCKER"))}}


def docker_destination(project: str) -> str | None:
    """Destination Docker Hub si la publication est activée (AGENT3_PUSH=1), sans aucun secret."""
    env = _agent3_env()
    if env.get("AGENT3_PUSH", "").lower() not in {"1", "true", "yes"}:
        return None
    account = env.get("DOCKER_IMAGE_NAMESPACE") or env.get("DOCKERHUB_USERNAME") or "?"
    return f"docker.io/{account}/{project.lower().replace(' ', '-')}"


def agent4_runs_dir() -> Path:
    return Path(os.environ.get("MONITORING_RUNS_DIR") or REPO / "agents" / "agent4" / "runs").expanduser()


# ---------------------------------------------------------------------------
# Lecture d'un run (rien de Slack ici : testable seul)
# ---------------------------------------------------------------------------


def _read(path: str | None) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}


def project_of(run: Run) -> str:
    spec = _read(run.state.get("artifacts", {}).get("project_spec"))
    return spec.get("project_name") or Path(run.state.get("project_dir", "?")).name


def gate_key(run: Run) -> str | None:
    q = run.question
    return f"{q['gate']}:{len(run.state.get('decisions', []))}" if q else None


def gate_details(run: Run, question: dict) -> list[str]:
    """Ce que l'humain doit voir pour décider, tiré des fichiers du contrat."""
    gate, lines = question["gate"], []
    f = _read(question.get("file_to_review"))
    if gate == "champion_replacement" and f:
        g = f.get("quality_gate") or {}
        metric = g.get("primary_metric", "score")
        lines.append(f"*{metric}* candidat *{g.get('value', 0):.4f}* · modèle naïf {g.get('baseline_value', 0):.4f}"
                     + (f" · champion actuel {g['champion_value']:.4f}" if g.get("champion_value") is not None else
                        " · pas encore de champion")
                     + f" · seuil {g.get('min_threshold', 0):.2f}")
        lines.append(f"Modèle : {f.get('params', {}).get('model', '?')} · {f.get('model_uri', '')}")
        dest = docker_destination(project_of(run))
        if dest:
            lines.append(f"Docker Hub : {dest} (dépôt public s'il n'existe pas)")
    elif gate == "retraining" and f:
        perf = f.get("current_performance") or {}
        lines.append(f"Raison : {f.get('reason', '?')} · modèle {f.get('model_uri', '?')}"
                     + (" · en production : " + ", ".join(f"{k} {v:.3f}" for k, v in list(perf.items())[:3])
                        if perf else ""))
        if f.get("diagnosis"):
            lines.append(f"> {f['diagnosis'][:600]}")
    elif gate == "manifest_confirmation" and f:
        lines.append(f"Tâche {f.get('task')} · cible `{f.get('target')}` · {f.get('primary_metric')} ≥ "
                     f"{f.get('min_threshold')} · confiance {f.get('confidence', 0):.2f}")
        if f.get("rationale"):
            lines.append(f"> {f['rationale'][:500]}")
    elif gate == "failure_review":
        lines.append(f"Étape : *{question.get('failed_step', '?')}*")
        lines += [f"• {r[:300]}" for r in question.get("reasons", [])[:3]]
        if question.get("supervisor_explanation"):
            lines.append(f"Superviseur : {question['supervisor_explanation'][:300]}")
    return lines


def gate_blocks(run: Run, settings: Settings) -> list[dict]:
    q = run.question
    gate = q["gate"]
    value = json.dumps({"run": run.run_id, "key": gate_key(run)})
    text = f"*{TITLES.get(gate, gate)}* · `{project_of(run)}`\n{q['question']}"
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
    details = gate_details(run, q)
    if details:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(details)[:2900]}})
    who = settings.approvers(gate)
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text":
                   f"run `{run.run_id}` · validation : "
                   + (", ".join(f"<@{u}>" for u in sorted(who)) if who else "tout le canal")}]})
    yes = "Relancer l'étape" if gate == "failure_review" else "Approuver"
    no = "Arrêter le run" if gate == "failure_review" else "Refuser"
    buttons = [{"type": "button", "action_id": APPROVE, "value": value, "style": "primary",
                "text": {"type": "plain_text", "text": yes}}]
    if gate == "champion_replacement" and docker_destination(project_of(run)):
        buttons[0]["text"]["text"] = "Approuver sans publier"
        buttons.insert(0, {"type": "button", "action_id": PUBLISH, "value": value, "style": "primary",
                           "text": {"type": "plain_text", "text": "Approuver et publier sur Docker Hub"}})
    buttons.append({"type": "button", "action_id": REJECT, "value": value, "style": "danger",
                    "text": {"type": "plain_text", "text": no}})
    blocks.append({"type": "actions", "elements": buttons})
    return blocks


def final_text(run: Run) -> str:
    s = run.state
    text = f"{FINAL_TEXT.get(run.status, run.status)} · `{project_of(run)}` · run `{run.run_id}`"
    for name, r in s.get("results", {}).items():
        if r["status"] in ("FAILED", "NEEDS_APPROVAL") or (r["status"] == "WARNING" and not r["can_proceed"]):
            text += "".join(f"\n• {name} : {reason[:200]}" for reason in r["reasons"][:2])
    return text


# ---------------------------------------------------------------------------
# Le service
# ---------------------------------------------------------------------------


class OrchestratorSlack:
    """Logique du service, testable sans Slack : `client` n'a besoin que de chat_postMessage et chat_update."""

    def __init__(self, client, settings: Settings, runs_dir: str = "runs", sync: bool = False):
        self.client, self.settings, self.runs_dir = client, settings, runs_dir
        self.sync = sync                    # tests : le travail s'exécute tout de suite, sans thread
        self.jobs: queue.Queue = queue.Queue()
        self.busy: set[str] = set()         # runs en cours d'exécution dans ce processus
        self.done: set[str] = set()         # runs terminés et annoncés : plus besoin de les relire
        self.triggered: set[str] = set()    # demandes de réentraînement déjà traitées (chemin:mtime)
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._tick_lock = threading.RLock()
        from .checkpoint import get_checkpointer
        self.saver = get_checkpointer(runs_dir)   # une seule connexion à la base pour tout le service

    # -- état posté, par run -----------------------------------------------------------------

    def _state(self, run_dir: Path) -> dict:
        return _read(str(run_dir / STATE_FILE)) if (run_dir / STATE_FILE).exists() else {}

    def _save(self, run_dir: Path, state: dict) -> None:
        (run_dir / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _post(self, state: dict, text: str, blocks: list[dict] | None = None, broadcast: bool = False) -> dict:
        kw = {"channel": self.settings.channel, "text": text[:3000]}
        if blocks:
            kw["blocks"] = blocks
        if state.get("thread_ts"):
            kw["thread_ts"] = state["thread_ts"]
            kw["reply_broadcast"] = broadcast
        return self.client.chat_postMessage(**kw)

    # -- un passage : gates à poster, fins de run, demandes de l'Agent 4 ----------------------

    def tick(self) -> None:
        with self._tick_lock:               # appelé par le thread de scrutation et après chaque exécution
            root = Path(self.runs_dir)
            if root.exists():
                for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime):
                    if d.name not in self.done and d.name not in self.busy:
                        self._visit(d)
            self._retraining_requests()

    def _visit(self, run_dir: Path) -> None:
        try:
            run = Run.load(run_dir.name, self.runs_dir, self.saver)
        except RunNotFound:                # dossier qui n'est pas un run (mlflow, check...)
            self.done.add(run_dir.name)
            return
        state = self._state(run_dir)
        if not state and run.status in FINAL and run_dir.stat().st_mtime < self.started_at:
            self.done.add(run.run_id)       # ancien run, terminé avant le service : on ne l'annonce pas
            return
        if not state.get("thread_ts"):
            msg = self._post({}, f":rocket: Pipeline sur `{project_of(run)}` · run `{run.run_id}`")
            state = {"thread_ts": msg["ts"], "gates": [], "final": None}
            self._save(run_dir, state)
        key = gate_key(run)
        if key and key not in state["gates"]:
            q = run.question
            msg = self._post(state, f"{TITLES.get(q['gate'], q['gate'])} : {project_of(run)}",
                             gate_blocks(run, self.settings), broadcast=True)
            state["gates"].append(key)
            state.setdefault("messages", {})[key] = msg.get("ts")
            self._save(run_dir, state)
        if run.status in FINAL:
            marker = f"{run.status}:{len(run.state.get('history', []))}"
            if state.get("final") != marker:
                self._post(state, final_text(run), broadcast=True)
                state["final"] = marker
                self._save(run_dir, state)
            self.done.add(run.run_id)

    # -- clic sur un bouton -------------------------------------------------------------------

    def on_click(self, action_id: str, value: str, user_id: str, user_name: str,
                 channel: str = "", ts: str = "") -> str:
        """Renvoie le message à montrer à la personne (éphémère)."""
        data = json.loads(value)
        try:
            run = Run.load(data["run"], self.runs_dir, self.saver)
        except RunNotFound:
            return f"Run {data['run']} introuvable (base différente de celle du service ?)."
        if data["run"] in self.busy or gate_key(run) != data["key"]:
            return "Cette validation a déjà été traitée."
        gate = run.question["gate"]
        if not self.settings.can(gate, user_id):
            return "Tu n'es pas dans la liste des personnes autorisées pour cette validation."
        if action_id == PUBLISH and not self.settings.can("docker_push", user_id):
            return "Tu n'es pas autorisé à publier sur Docker Hub : choisis « Approuver sans publier »."
        decision = "reject" if action_id == REJECT else "approve"
        if gate == "champion_replacement" and decision == "approve":
            (run.run_dir / PUBLISH_FILE).write_text(json.dumps({"push": action_id == PUBLISH, "by": user_name}),
                                                    encoding="utf-8")
        verdict = {APPROVE: "approuvé", PUBLISH: "approuvé, avec publication Docker Hub",
                   REJECT: "refusé"}[action_id]
        if channel and ts:                  # les boutons disparaissent : la décision est visible de tous
            blocks = [b for b in gate_blocks(run, self.settings) if b["type"] != "actions"]
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text":
                           f"<@{user_id}> a {verdict} · le pipeline continue"}]})
            self.client.chat_update(channel=channel, ts=ts, text=f"{gate} : {verdict}", blocks=blocks)
        self._submit(run.run_id, lambda: Run.load(run.run_id, self.runs_dir, self.saver).answer(
            {"decision": decision, "by": f"{user_name} (Slack)", "comment": verdict}))
        return f"Décision enregistrée : {verdict}."

    # -- demandes de réentraînement de la gateway de l'Agent 4 ------------------------------

    def _retraining_requests(self) -> None:
        root = agent4_runs_dir()
        if not root.exists():
            return
        for req in root.glob("*/actions/retraining_request.json"):
            seen = f"{req}:{req.stat().st_mtime}"
            if seen in self.triggered:      # une seule fois par demande (l'Agent 4 la marque remise)
                continue
            project = req.parent.parent.name
            run = self._latest_run(project)
            if run is not None and (run.run_id in self.busy or run.status == "WAITING_HUMAN"):
                continue                    # déjà en cours : la demande sera vue au prochain passage
            self.triggered.add(seen)
            if _read(str(req.parent.parent / "monitoring.json")).get("project") != project:
                continue                    # archive (04_x.old...) : pas un monitoring actif
            if run is None or not self._deployed(run):
                self._post({}, f":warning: L'Agent 4 demande un réentraînement de `{project}`, mais aucun run "
                               "déployé de ce projet n'est enregistré : lancer d'abord le pipeline.")
                continue
            asked = _read(str(req)).get("model_uri")
            served = _read(run.state.get("artifacts", {}).get("deployment_result")).get("model_uri")
            if asked != served:             # demande ancienne ou d'un autre déploiement : jamais de réentraînement
                self._post({}, f":information_source: Demande de réentraînement de `{project}` ignorée : elle "
                               f"concerne `{asked}`, le run `{run.run_id}` sert `{served}`.")
                continue
            self.done.discard(run.run_id)   # le run va repartir : on le suit de nouveau
            self._post(self._state(run.run_dir), f":chart_with_downwards_trend: L'Agent 4 demande un réentraînement "
                                                 f"de `{project}` : relance du monitoring (run `{run.run_id}`)",
                       broadcast=True)
            self._submit(run.run_id, lambda r=run: Run.load(r.run_id, self.runs_dir, self.saver).monitor())

    @staticmethod
    def _deployed(run: Run) -> bool:
        ok = run.status in ("DONE", "CHAMPION_KEPT") or (run.status == "FAILED" and run.state.get("retrain_count", 0))
        return bool(ok and "agent4" in run.state.get("results", {}))

    def _latest_run(self, project: str) -> Run | None:
        """Le run le plus récent de ce projet qui a atteint l'Agent 4 (ou qui attend une décision)."""
        root = Path(self.runs_dir)
        if not root.exists():
            return None
        for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                run = Run.load(d.name, self.runs_dir, self.saver)
            except RunNotFound:            # dossier qui n'est pas un run (mlflow, check...)
                continue
            if project_of(run) == project and ("agent4" in run.state.get("results", {})
                                               or run.status == "WAITING_HUMAN"):
                return run
        return None

    # -- exécution (un run à la fois : les agents sont lourds) --------------------------------

    def _submit(self, run_id: str, work: Callable[[], None]) -> None:
        self.busy.add(run_id)
        if self.sync:
            self._execute(run_id, work)
        else:
            self.jobs.put((run_id, work))

    def _execute(self, run_id: str, work: Callable[[], None]) -> None:
        try:
            work()
        except Exception as exc:
            log.exception("run %s", run_id)
            self._post({}, f":x: run `{run_id}` : erreur pendant l'exécution ({type(exc).__name__}: {exc})"[:500])
        finally:
            self.busy.discard(run_id)
            self.done.discard(run_id)
            self.tick()

    def worker(self) -> None:
        while True:
            run_id, work = self.jobs.get()
            with self._lock:
                self._execute(run_id, work)


def serve(runs_dir: str = "runs") -> int:
    """Lance le service (bloquant)."""
    settings = Settings()
    if not settings.configured:
        print("Slack non configuré : ORCH_SLACK_BOT_TOKEN, ORCH_SLACK_APP_TOKEN et ORCH_SLACK_CHANNEL "
              "dans le .env racine (voir docs/slack.md).")
        return 2
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = App(token=settings.bot_token)
    service = OrchestratorSlack(app.client, settings, runs_dir)

    def click(ack, body, respond, action):
        ack()
        user = body.get("user", {})
        msg = service.on_click(action["action_id"], action["value"], user.get("id", ""),
                               user.get("name") or user.get("username") or user.get("id", "?"),
                               body.get("channel", {}).get("id", ""), body.get("container", {}).get("message_ts", ""))
        respond(response_type="ephemeral", replace_original=False, text=msg)

    for action_id in (APPROVE, PUBLISH, REJECT):
        app.action(action_id)(click)

    threading.Thread(target=service.worker, daemon=True, name="runs").start()

    def poll():
        while True:
            if not service.busy:
                try:
                    service.tick()
                except Exception:
                    log.exception("tick")
            time.sleep(settings.poll_s)

    threading.Thread(target=poll, daemon=True, name="poll").start()
    print(f"Slack : canal {settings.channel} · validations : "
          + (", ".join(f"{g}={len(u)}" for g, u in settings.per_gate.items()) or "liste commune")
          + f" · défaut : {len(settings.default) or 'tout le canal'}")
    print("En attente des runs (Ctrl+C pour arrêter)…")
    SocketModeHandler(app, settings.app_token).start()
    return 0
