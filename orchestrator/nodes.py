"""
Les nœuds du graphe : le validateur (réel), les agents (stubs pour l'instant)
et les nœuds de fin.

Un nœud reçoit l'état et renvoie SEULEMENT ce qu'il veut modifier dans l'état.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from common.schemas import AgentInput, AgentResult, AgentStatus, ValidatorHandoff

from .state import PipelineState

VALIDATOR_TIMEOUT_S = 900

# statut du handoff -> statut d'AgentResult
_HANDOFF_TO_STATUS = {
    "READY": AgentStatus.OK,
    "READY_WITH_WARNINGS": AgentStatus.WARNING,
    "AWAITING_CONFIRMATION": AgentStatus.NEEDS_APPROVAL,
    "INCOMPLETE": AgentStatus.FAILED,
    "BLOCKED": AgentStatus.FAILED,
}


# ---------------------------------------------------------------------------
# Validateur (le seul composant réel pour l'instant)
# ---------------------------------------------------------------------------


def run_validator(project_dir: str, report: Path, snapshot_root: Path) -> subprocess.CompletedProcess:
    """Lance le validateur dans un processus séparé (codes de sortie : 0, 10 ou 20)."""
    cmd = [
        sys.executable, "-m", "validator", project_dir,
        "--json", str(report),
        "--snapshot-dir", str(snapshot_root),
        "-q",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=VALIDATOR_TIMEOUT_S)


def verify(report: Path, project_dir: str) -> list[str]:
    """Contrôle d'intégrité fourni par le validateur. Renvoie les raisons de refus."""
    from validator.handoff import verify_handoff

    check = verify_handoff(str(report), project_dir)
    return [] if check.ok else list(check.reasons)


def validate(state: PipelineState) -> dict:
    run_dir = Path(state["run_dir"])
    out = run_dir / "validator"
    out.mkdir(parents=True, exist_ok=True)
    report = out / "compatibility_report.json"
    started = time.monotonic()

    proc = run_validator(state["project_dir"], report, run_dir / "snapshot")

    if not report.exists():
        result = AgentResult(
            agent="validator",
            status=AgentStatus.FAILED,
            can_proceed=False,
            reasons=[f"le validateur n'a pas produit de rapport (code {proc.returncode})",
                     proc.stderr.strip()[-500:]],
        )
        return {"handoff_status": "ERROR", **_record("validator", result, started)}

    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        handoff = ValidatorHandoff(**data["handoff"])
    except Exception as exc:  # rapport illisible : on s'arrête proprement
        result = AgentResult(
            agent="validator",
            status=AgentStatus.FAILED,
            can_proceed=False,
            outputs={"compatibility_report": str(report)},
            reasons=[f"rapport du validateur illisible : {type(exc).__name__}: {exc}"[:500]],
        )
        return {"handoff_status": "ERROR", **_record("validator", result, started)}
    status = _HANDOFF_TO_STATUS[handoff.status.value]
    reasons = [handoff.reason] if handoff.reason else []
    can_proceed = handoff.can_proceed

    if can_proceed:
        refused = verify(report, state["project_dir"])
        if refused:
            status, can_proceed = AgentStatus.FAILED, False
            reasons += refused

    result = AgentResult(
        agent="validator",
        status=status,
        can_proceed=can_proceed,
        outputs={"compatibility_report": str(report)},
        reasons=reasons,
        warnings=[c.message for c in handoff.carry_forward],
    )
    return {
        "report_path": str(report),
        # les agents lisent la copie figée ; à défaut (pas de snapshot), le projet d'origine
        "snapshot_dir": data.get("snapshot_path") or state["project_dir"],
        "handoff_status": handoff.status.value,
        "artifacts": result.outputs,
        **_record("validator", result, started),
    }


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def make_agent_node(node_name: str, agent_fn: Callable[[AgentInput], AgentResult]):
    """Enveloppe commune à tous les agents : prépare l'entrée, mesure le temps,
    transforme toute exception en FAILED, vérifie que la réponse respecte le schéma."""

    def node(state: PipelineState) -> dict:
        workdir = Path(state["run_dir"]) / node_name
        workdir.mkdir(parents=True, exist_ok=True)
        inp = AgentInput(
            run_id=state["run_id"],
            project_dir=state["snapshot_dir"],
            report_path=state["report_path"],
            workdir=str(workdir),
            inputs=dict(state.get("artifacts", {})),
        )
        started = time.monotonic()
        try:
            result = AgentResult.model_validate(agent_fn(inp).model_dump())
        except Exception as exc:  # un agent qui plante ne doit pas faire planter le pipeline
            result = AgentResult(
                agent=_agent_name(node_name),
                status=AgentStatus.FAILED,
                can_proceed=False,
                reasons=[f"{type(exc).__name__}: {exc}"],
            )
        return {"artifacts": result.outputs, **_record(node_name, result, started)}

    node.__name__ = node_name
    return node


def _agent_name(node_name: str) -> str:
    return "agent1" if node_name == "propose_manifest" else node_name


def _record(node_name: str, result: AgentResult, started: float) -> dict:
    """Mise à jour de l'état : dernier résultat de l'étape + une ligne d'historique."""
    dumped = _dump(result, started)
    return {"results": {node_name: dumped}, "history": [{"node": node_name, **dumped}]}


def _dump(result: AgentResult, started: float) -> dict:
    result.duration_s = round(time.monotonic() - started, 3)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Fins de pipeline
# ---------------------------------------------------------------------------


def make_end(final_status: str):
    def node(state: PipelineState) -> dict:
        return {"final_status": final_status}

    node.__name__ = f"end_{final_status.lower()}"
    return node
