"""Point d'entrée du workflow global. NON IMPLÉMENTÉ volontairement."""
from __future__ import annotations

from .state import WorkflowState


def run_workflow(state: WorkflowState, client) -> WorkflowState:
    raise NotImplementedError("orchestrateur non implémenté : voir routing.py pour le contrat de routage")
