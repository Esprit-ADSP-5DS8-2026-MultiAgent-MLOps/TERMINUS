"""
Le graphe du pipeline et la fonction qui l'exécute.

Les agents utilisés (vrais ou stubs) viennent de registry.py : pour brancher un vrai agent,
on ajoute une ligne dans registry.REAL, rien à changer ici.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .checkpoint import get_checkpointer
from .gates import make_gate
from .llm import default_mode, get_decider
from .nodes import make_agent_node, make_end, make_repair_node, validate
from .state import PipelineState, initial_state
from .supervisor import DESTINATIONS, supervisor
from .registry import resolve
from .stubs.agents import (  # noqa: F401 - réexportés pour les tests (graph.agent3_stub)
    SIMULATE_FILE,
    agent1_stub,
    agent2_stub,
    agent3_stub,
    agent4_stub,
    propose_manifest_stub,
)

# Vrais agents intégrés, stubs pour les autres (ORCH_AGENTS : real | stub | liste)
AGENTS, AGENT_ORIGIN = resolve()

END_STATUSES = ["DONE", "BLOCKED", "FAILED", "REJECTED", "MANIFEST_PROPOSED", "CHAMPION_KEPT"]

# Vrais agents utilisés par un run, notés au lancement : un run repris (resume, approve) garde les
# mêmes agents, même si ORCH_AGENTS a changé entre-temps dans le terminal.
AGENTS_FILE = "agents.json"


def _real(origin: dict[str, str]) -> list[str]:
    return sorted(name for name, o in origin.items() if o == "réel")


def agents_for(run_dir: Path) -> dict[str, Callable] | None:
    """Les agents d'un run existant, s'ils diffèrent de ceux du processus actuel (sinon None)."""
    f = run_dir / AGENTS_FILE
    if not f.exists():
        return None
    saved = json.loads(f.read_text())["real"]
    if saved == _real(AGENT_ORIGIN):
        return None
    agents, _ = resolve(set(saved))
    return agents


WORK_NODES = ["validate", *AGENTS, "gate_manifest", "gate_champion", "gate_retraining", "gate_failure"]


def build_graph(checkpointer=None, agents: dict[str, Callable] | None = None):
    """Chaque étape renvoie vers le superviseur, qui choisit la suivante (supervisor.py)."""
    g = StateGraph(PipelineState)

    # étapes
    g.add_node("validate", validate)
    for name, fn in (agents or AGENTS).items():
        g.add_node(name, make_repair_node(fn) if name == "repair" else make_agent_node(name, fn))
    g.add_node("gate_manifest", make_gate("manifest_confirmation", "proposed_manifest", "gate_manifest"))
    g.add_node("gate_champion", make_gate("champion_replacement", "mlflow_metadata", "gate_champion"))
    g.add_node("gate_retraining", make_gate("retraining", "retraining_request", "gate_retraining"))
    g.add_node("gate_failure", make_gate("failure_review", None, "gate_failure"))
    for status in END_STATUSES:
        g.add_node(f"end_{status.lower()}", make_end(status))
        g.add_edge(f"end_{status.lower()}", END)

    # le superviseur décide de la suite après chaque étape
    g.add_node("supervisor", supervisor, destinations=DESTINATIONS)
    g.add_edge(START, "validate")
    for node in WORK_NODES:
        g.add_edge(node, "supervisor")

    return g.compile(checkpointer=checkpointer or MemorySaver())


Decider = Callable[[dict], dict]  # reçoit la question d'une gate, renvoie la décision


class RunNotFound(Exception):
    pass


class Run:
    """Un run : démarre, s'arrête aux gates humaines, reprend.

    L'état est sauvegardé après chaque étape (SQLite ou PostgreSQL, voir checkpoint.py) :
    un run peut être rechargé plus tard avec Run.load(run_id), même après
    avoir fermé le terminal ou après un plantage.

    Utilisé par la ligne de commande et par l'interface Streamlit.
    """

    def __init__(self, run_id: str, runs_dir: str = "runs", checkpointer=None):
        self.run_id = run_id
        self.runs_dir = Path(runs_dir).resolve()
        self.run_dir = self.runs_dir / run_id
        self.graph = build_graph(checkpointer or get_checkpointer(self.runs_dir), agents_for(self.run_dir))
        self.config = {"configurable": {"thread_id": run_id}}

    # -- création / chargement ------------------------------------------------

    @classmethod
    def start(cls, project_dir: str, runs_dir: str = "runs", supervisor: str | None = None,
              simulate: dict | None = None) -> "Run":
        """Crée un nouveau run et l'exécute jusqu'à la première pause (ou la fin).

        supervisor : "rules" ou "gemini" ; par défaut ORCH_SUPERVISOR, sinon gemini si
                     GEMINI_API_KEY est définie, sinon rules (voir llm.default_mode).
        simulate   : pannes simulées pour les démos, enregistrées avec le run
                     ({"drift": bool, "failure": "transient" | "permanent" | None}).
        """
        mode = supervisor or default_mode()
        get_decider(mode)  # erreur claire tout de suite si le mode est inconnu
        run = cls(uuid.uuid4().hex[:12], runs_dir)
        run.run_dir.mkdir(parents=True, exist_ok=True)
        if simulate is None:  # par défaut, lues dans l'environnement
            simulate = {
                "drift": os.environ.get("ORCH_SIMULATE_DRIFT") == "1",
                "failure": os.environ.get("ORCH_SIMULATE_FAILURE") or None,
            }
        (run.run_dir / SIMULATE_FILE).write_text(json.dumps(simulate))
        (run.run_dir / AGENTS_FILE).write_text(json.dumps({"real": _real(AGENT_ORIGIN)}))
        run.graph.invoke(
            initial_state(run.run_id, str(Path(project_dir).resolve()), str(run.run_dir), mode),
            run.config,
        )
        run._save_if_finished()
        return run

    @classmethod
    def load(cls, run_id: str, runs_dir: str = "runs", checkpointer=None) -> "Run":
        """Recharge un run existant depuis la base SQLite."""
        run = cls(run_id, runs_dir, checkpointer)
        if not run.state:
            raise RunNotFound(f"aucun run '{run_id}' dans {run.runs_dir}")
        return run

    # -- lecture de l'état ---------------------------------------------------

    @property
    def snapshot(self):
        return self.graph.get_state(self.config)

    @property
    def state(self) -> dict:
        return dict(self.snapshot.values)

    @property
    def question(self) -> dict | None:
        """La question de la gate humaine en attente, ou None."""
        interrupts = self.snapshot.interrupts
        return interrupts[0].value if interrupts else None

    @property
    def status(self) -> str:
        """DONE/BLOCKED/... si terminé, WAITING_HUMAN si en pause,
        INTERRUPTED si arrêté au milieu (plantage) et donc reprenable."""
        snap = self.snapshot
        if snap.values.get("final_status"):
            return snap.values["final_status"]
        if snap.interrupts:
            return "WAITING_HUMAN"
        if snap.next:
            return "INTERRUPTED"
        return "UNKNOWN"

    @property
    def finished(self) -> bool:
        return bool(self.state.get("final_status"))

    # -- actions --------------------------------------------------------------

    def answer(self, decision: dict) -> None:
        """Répond à la gate en attente et laisse le pipeline continuer."""
        if self.question is None:
            raise RuntimeError(f"le run {self.run_id} n'attend pas de décision ({self.status})")
        self.graph.invoke(Command(resume=decision), self.config)
        self._save_if_finished()

    def resume(self) -> None:
        """Reprend un run arrêté au milieu (plantage), à l'étape où il s'était arrêté."""
        if self.status != "INTERRUPTED":
            raise RuntimeError(f"le run {self.run_id} n'est pas à reprendre ({self.status})")
        self.graph.invoke(None, self.config)
        self._save_if_finished()

    def monitor(self) -> None:
        """Run déployé : relance l'Agent 4 sur le modèle en production.

        Si le monitoring demande un réentraînement (retraining_request), le run repart :
        gate « retraining » -> Agent 2 -> gate « champion_replacement » -> Agent 3 -> Agent 4.
        Sinon il se termine de nouveau en DONE. Un candidat refusé (quality gate ou champion meilleur) termine
        en CHAMPION_KEPT : le modèle déployé continue de servir et reste surveillé. Chaque appel est un nouvel épisode : le compteur
        de réentraînements repart à zéro (le garde-fou évite seulement une boucle dans un épisode).
        """
        deployed = self.status in ("DONE", "CHAMPION_KEPT") or (
            self.status == "FAILED" and self.state.get("retrain_count", 0))   # échec pendant un réentraînement
        if not deployed or "agent4" not in self.state.get("results", {}):
            raise RuntimeError(f"le run {self.run_id} n'a pas de modèle déployé et surveillé ({self.status}) : "
                               "monitor s'applique à un run déployé (DONE, ou CHAMPION_KEPT après un réentraînement)")
        self.graph.update_state(self.config, {"final_status": "", "retrain_count": 0, "last_step": "agent3"},
                                as_node="agent3")
        self.graph.invoke(None, self.config)
        self._save_if_finished()

    def summary(self) -> dict:
        return self.state

    def _save_if_finished(self) -> None:
        if self.finished:
            (self.run_dir / "run_summary.json").write_text(
                json.dumps(self.summary(), indent=2, ensure_ascii=False)
            )


def list_runs(runs_dir: str = "runs") -> list[Run]:
    """Tous les runs enregistrés, du plus récent au plus ancien."""
    root = Path(runs_dir).resolve()
    if not root.exists():
        return []
    dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    saver = get_checkpointer(root)  # une seule connexion pour tous les runs
    runs = []
    for d in dirs:
        try:
            runs.append(Run.load(d.name, runs_dir, saver))
        except RunNotFound:
            continue
    return runs


def run_pipeline(project_dir: str, decide: Decider, runs_dir: str = "runs",
                 supervisor: str | None = None, simulate: dict | None = None) -> dict:
    """Exécute le pipeline de bout en bout. `decide` répond aux gates humaines."""
    run = Run.start(project_dir, runs_dir, supervisor=supervisor, simulate=simulate)
    while not run.finished:
        run.answer(decide(run.question))
    return run.summary()
