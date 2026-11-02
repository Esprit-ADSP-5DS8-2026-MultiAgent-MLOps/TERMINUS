"""
Gates humaines : le pipeline se met en pause (interrupt) et attend une décision.

La décision attendue à la reprise est un dict :
    {"decision": "approve" | "reject", "by": "Nom", "comment": "..."}
"""

from __future__ import annotations

from langgraph.types import interrupt

from common.schemas import HumanDecision

from .state import PipelineState

MAX_RETRAINS = 1  # garde-fou contre une boucle de réentraînement infinie

QUESTIONS = {
    "manifest_confirmation": "Le projet n'a pas de mlops.yaml. Valider le manifeste proposé ?",
    "champion_replacement": "Le candidat a passé le quality gate. Le promouvoir en champion ?",
    "retraining": "Un drift a été détecté. Lancer un réentraînement ?",
    "failure_review": "Une étape a échoué. La relancer ?",
    "repair_review": "La réparation touche un périmètre sensible. Valider la copie réparée ?",
    "validator_override": "Le validateur v2 ne confirme pas le blocage du v1. Passer outre le v1 ?",
}


def make_gate(gate: str, artifact_key: str | None, node_name: str):
    """Crée un nœud qui suspend le pipeline et enregistre la décision humaine."""

    def node(state: PipelineState) -> dict:
        # Tout ce qui est avant interrupt() est rejoué à la reprise : rien d'irréversible ici.
        question = {
            "gate": gate,
            "question": QUESTIONS[gate],
            "run_id": state["run_id"],
            "file_to_review": state.get("artifacts", {}).get(artifact_key) if artifact_key else None,
        }
        if gate == "failure_review":  # l'humain doit voir quelle étape a échoué et pourquoi
            step = state.get("failed_step", "")
            result = state.get("results", {}).get(step, {})
            question["failed_step"] = step
            question["reasons"] = result.get("reasons", [])
            explanations = [s for s in state.get("supervision", []) if s.get("action") == "gate_failure"]
            if explanations:
                question["supervisor_explanation"] = explanations[-1]["explanation"]
        if gate in ("repair_review", "validator_override"):  # ce que la réparation a vu et fait
            question["reasons"] = state.get("results", {}).get("repair", {}).get("reasons", [])
        answer = interrupt(question)
        decision = HumanDecision(gate=gate, **answer)
        update: dict = {"decisions": [decision.model_dump(mode="json")], "last_step": node_name}
        if gate == "retraining" and decision.decision == "approve":
            update["retrain_count"] = state.get("retrain_count", 0) + 1
        return update

    node.__name__ = f"gate_{gate}"
    return node


def last_decision(state: PipelineState) -> str:
    return state["decisions"][-1]["decision"]
