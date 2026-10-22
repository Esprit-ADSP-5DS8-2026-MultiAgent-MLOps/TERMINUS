"""
Le graphe du pipeline et la fonction qui l'exécute.

Pour brancher un vrai agent, remplace la fonction dans AGENTS, par exemple :
    from agents.agent1_analysis.agent import run as agent1_run
    AGENTS["agent1"] = agent1_run
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from . import routing
from .checkpoint import get_checkpointer
from .gates import make_gate
from .nodes import make_agent_node, make_end, validate
from .state import PipelineState, initial_state
from .stubs.agents import (
    agent1_stub,
    agent2_stub,
    agent3_stub,
    agent4_stub,
    propose_manifest_stub,
)

AGENTS = {
    "propose_manifest": propose_manifest_stub,
    "agent1": agent1_stub,
    "agent2": agent2_stub,
    "agent3": agent3_stub,
    "agent4": agent4_stub,
}

END_STATUSES = ["DONE", "BLOCKED", "FAILED", "REJECTED", "MANIFEST_PROPOSED"]


def build_graph(checkpointer=None):
    g = StateGraph(PipelineState)

    # nœuds
    g.add_node("validate", validate)
    for name, fn in AGENTS.items():
        g.add_node(name, make_agent_node(name, fn))
    g.add_node("gate_manifest", make_gate("manifest_confirmation", "proposed_manifest"))
    g.add_node("gate_champion", make_gate("champion_replacement", "mlflow_metadata"))
    g.add_node("gate_retraining", make_gate("retraining", "retraining_request"))
    for status in END_STATUSES:
        g.add_node(f"end_{status.lower()}", make_end(status))
        g.add_edge(f"end_{status.lower()}", END)

    # transitions
    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", routing.after_validate,
                            ["agent1", "propose_manifest", "end_blocked", "end_failed"])
    g.add_conditional_edges("propose_manifest", routing.after_propose_manifest,
                            ["gate_manifest", "end_failed"])
    g.add_conditional_edges("gate_manifest", routing.after_gate_manifest,
                            ["end_manifest_proposed", "end_rejected"])
    g.add_conditional_edges("agent1", routing.after_agent1, ["agent2", "end_failed"])
    g.add_conditional_edges("agent2", routing.after_agent2,
                            ["gate_champion", "agent3", "end_failed"])
    g.add_conditional_edges("gate_champion", routing.after_gate_champion,
                            ["agent3", "end_rejected"])
    g.add_conditional_edges("agent3", routing.after_agent3, ["agent4", "end_failed"])
    g.add_conditional_edges("agent4", routing.after_agent4,
                            ["gate_retraining", "end_done", "end_failed"])
    g.add_conditional_edges("gate_retraining", routing.after_gate_retraining,
                            ["agent2", "end_done"])

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
        self.graph = build_graph(checkpointer or get_checkpointer(self.runs_dir))
        self.config = {"configurable": {"thread_id": run_id}}

    # -- création / chargement ------------------------------------------------

    @classmethod
    def start(cls, project_dir: str, runs_dir: str = "runs") -> "Run":
        """Crée un nouveau run et l'exécute jusqu'à la première pause (ou la fin)."""
        run = cls(uuid.uuid4().hex[:12], runs_dir)
        run.run_dir.mkdir(parents=True, exist_ok=True)
        run.graph.invoke(
            initial_state(run.run_id, str(Path(project_dir).resolve()), str(run.run_dir)),
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


def run_pipeline(project_dir: str, decide: Decider, runs_dir: str = "runs") -> dict:
    """Exécute le pipeline de bout en bout. `decide` répond aux gates humaines."""
    run = Run.start(project_dir, runs_dir)
    while not run.finished:
        run.answer(decide(run.question))
    return run.summary()
