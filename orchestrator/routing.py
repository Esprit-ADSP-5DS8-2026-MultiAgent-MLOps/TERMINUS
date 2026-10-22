"""
Décisions de routage : après chaque étape, quelle est la suivante ?

Ces fonctions ne modifient rien : elles lisent l'état et renvoient le nom du nœud suivant.
"""

from __future__ import annotations

from .gates import MAX_RETRAINS, last_decision
from .state import PipelineState


def _result(state: PipelineState, node: str) -> dict:
    return state["results"][node]


def after_validate(state: PipelineState) -> str:
    r = _result(state, "validator")
    if r["can_proceed"]:
        return "agent1"
    if state.get("handoff_status") == "AWAITING_CONFIRMATION":
        return "propose_manifest"
    if state.get("handoff_status") == "ERROR":  # panne du validateur, pas un projet rejeté
        return "end_failed"
    return "end_blocked"


def after_propose_manifest(state: PipelineState) -> str:
    r = _result(state, "propose_manifest")
    return "gate_manifest" if r["status"] == "NEEDS_APPROVAL" else "end_failed"


def after_gate_manifest(state: PipelineState) -> str:
    # L'orchestrateur ne modifie jamais le projet : on s'arrête ici,
    # l'utilisateur ajoute le mlops.yaml validé puis relance.
    return "end_manifest_proposed" if last_decision(state) == "approve" else "end_rejected"


def after_agent1(state: PipelineState) -> str:
    return "agent2" if _result(state, "agent1")["can_proceed"] else "end_failed"


def after_agent2(state: PipelineState) -> str:
    r = _result(state, "agent2")
    if r["status"] == "NEEDS_APPROVAL":
        return "gate_champion"
    return "agent3" if r["can_proceed"] else "end_failed"


def after_gate_champion(state: PipelineState) -> str:
    return "agent3" if last_decision(state) == "approve" else "end_rejected"


def after_agent3(state: PipelineState) -> str:
    return "agent4" if _result(state, "agent3")["can_proceed"] else "end_failed"


def after_agent4(state: PipelineState) -> str:
    r = _result(state, "agent4")
    if not r["can_proceed"]:
        return "end_failed"
    if "retraining_request" in r["outputs"] and state.get("retrain_count", 0) < MAX_RETRAINS:
        return "gate_retraining"
    return "end_done"


def after_gate_retraining(state: PipelineState) -> str:
    # Réentraînement approuvé : le candidat repasse par Agent 2 (quality gate) puis Agent 3.
    return "agent2" if last_decision(state) == "approve" else "end_done"
