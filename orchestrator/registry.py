"""
Quels agents tournent : les vrais quand ils sont intégrés, sinon les stubs.

Chaque vrai agent est déclaré par le chemin de sa fonction `run(AgentInput) -> AgentResult`
(contrat : common/schemas.py). On l'importe seulement au moment de construire le graphe :
un agent qui manque ou qui ne s'importe pas n'empêche pas le pipeline de tourner, son stub prend la place
et le run le signale.

Choisir le mode avec la variable ORCH_AGENTS :
    ORCH_AGENTS=real    (défaut) vrais agents intégrés, stubs pour les autres
    ORCH_AGENTS=stub    tous les stubs (tests de l'orchestrateur, démos sans infrastructure)
    ORCH_AGENTS=agent4  liste explicite des vrais agents à utiliser (séparés par des virgules)
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable

from .stubs.agents import (
    agent1_stub,
    agent2_stub,
    agent3_stub,
    agent4_stub,
    propose_manifest_stub,
    repair_stub,
)

log = logging.getLogger("orchestrator.registry")

STUBS: dict[str, Callable] = {
    "propose_manifest": propose_manifest_stub,
    "agent1": agent1_stub,
    "agent2": agent2_stub,
    "agent3": agent3_stub,
    "agent4": agent4_stub,
    "repair": repair_stub,
}

# Agents intégrés : on ajoute une ligne à chaque intégration (voir docs/integration.md).
# Avant d'ajouter la ligne : python -m orchestrator check-agent <nom> <module:fonction>
REAL: dict[str, str] = {
    "agent4": "monitoring_agent.pipeline:run",       # agents/agent4 (monitoring)
    "agent1": "agents.agent1.orchestrator_adapter:run",              # agents/agent1 (analyse)
    "propose_manifest": "agents.agent1.orchestrator_adapter:propose_manifest",
    "agent2": "mlflow_agent.orchestrator_adapter:run",               # agents/agent2 (MLflow)
    "agent3": "agent_api_docker.orchestrator_adapter:run",           # agents/agent3 (API + Docker)
    "repair": "repair_agent.orchestrator_adapter:run",               # agents/validator (Validator v2, Repair, Judge)
}


def _load(path: str) -> Callable:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def wanted() -> set[str]:
    mode = os.environ.get("ORCH_AGENTS", "real").strip().lower()
    if mode == "real":
        return set(REAL)
    if mode == "stub":
        return set()
    return {a.strip() for a in mode.split(",") if a.strip()}


def resolve(names: set[str] | None = None) -> tuple[dict[str, Callable], dict[str, str]]:
    """Renvoie (agents utilisés, origine de chacun : "réel", "stub" ou "stub (raison)").

    names : vrais agents voulus ; par défaut ceux de ORCH_AGENTS (voir wanted()).
    """
    agents, origin = dict(STUBS), {name: "stub" for name in STUBS}
    for name in (wanted() if names is None else names):
        if name not in REAL:
            origin[name] = "stub (pas encore intégré)"
            continue
        try:
            agents[name] = _load(REAL[name])
            origin[name] = "réel"
        except Exception as exc:  # noqa: BLE001 - paquet non installé, erreur d'import : on garde le stub
            log.warning("agent %s indisponible, stub utilisé : %s", name, exc)
            origin[name] = f"stub ({type(exc).__name__}: {exc})"[:200]
    return agents, origin
