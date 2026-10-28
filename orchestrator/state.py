"""
État partagé du pipeline (ce que LangGraph sauvegarde après chaque étape).

On n'y met que des petites informations : identifiants, chemins, statuts.
Jamais les données elles-mêmes.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def merge_dicts(old: dict | None, new: dict | None) -> dict:
    """Réducteur : chaque étape AJOUTE ses entrées au dictionnaire au lieu de l'écraser."""
    return {**(old or {}), **(new or {})}


class PipelineState(TypedDict, total=False):
    run_id: str
    project_dir: str        # projet d'origine, donné au lancement
    run_dir: str            # runs/<run_id>/ : tout ce que produit ce run
    report_path: str        # compatibility_report.json de CE run
    snapshot_dir: str       # copie figée du projet validé (ce que lisent les agents)
    handoff_status: str     # READY, AWAITING_CONFIRMATION, BLOCKED...

    # nom de sortie -> chemin du fichier (ex. "project_spec" -> ".../project_spec.json")
    artifacts: Annotated[dict[str, str], merge_dicts]
    # nom du nœud -> AgentResult (sous forme de dict)
    results: Annotated[dict[str, dict], merge_dicts]
    # décisions humaines, dans l'ordre (HumanDecision sous forme de dict)
    decisions: Annotated[list[dict], operator.add]
    # chaque exécution d'étape, dans l'ordre (results ne garde que la dernière par étape)
    history: Annotated[list[dict], operator.add]

    retrain_count: int
    final_status: str       # DONE, BLOCKED, FAILED, REJECTED_BY_HUMAN, MANIFEST_PROPOSED

    # --- superviseur -----------------------------------------------------------
    supervisor_mode: str    # "rules" (sans LLM) ou "gemini"
    last_step: str          # dernière étape terminée (le superviseur décide de la suite)
    # chaque décision du superviseur : action choisie, explication, qui a décidé
    supervision: Annotated[list[dict], operator.add]
    # nombre de nouvelles tentatives par étape (limite les boucles)
    attempts: Annotated[dict[str, int], merge_dicts]
    failed_step: str        # étape en échec soumise à un humain (gate failure_review)


def initial_state(run_id: str, project_dir: str, run_dir: str,
                  supervisor_mode: str = "rules") -> PipelineState:
    return {
        "run_id": run_id,
        "project_dir": project_dir,
        "run_dir": run_dir,
        "artifacts": {},
        "results": {},
        "decisions": [],
        "history": [],
        "retrain_count": 0,
        "supervisor_mode": supervisor_mode,
        "supervision": [],
        "attempts": {},
    }
