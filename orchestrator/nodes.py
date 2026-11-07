"""
Les nœuds du graphe : le validateur, les agents (vrais ou stubs, voir registry.py),
la réparation et les nœuds de fin.

Un nœud reçoit l'état et renvoie SEULEMENT ce qu'il veut modifier dans l'état.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    RepairResult,
    ValidatorHandoff,
)

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
# Validateur
# ---------------------------------------------------------------------------


def run_validator(project_dir: str, report: Path, snapshot_root: Path, waive: list[str] | None = None,
                  waived_by: str | None = None) -> subprocess.CompletedProcess:
    """Lance le validateur dans un processus séparé (codes de sortie : 0, 10 ou 20)."""
    cmd = [
        sys.executable, "-m", "validator", project_dir,
        "--json", str(report),
        "--snapshot-dir", str(snapshot_root),
        "-q",
    ]
    for code in waive or []:
        cmd += ["--waive", code]
    if waive and waived_by:
        cmd += ["--waived-by", waived_by]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=VALIDATOR_TIMEOUT_S)


def verify(report: Path, project_dir: str) -> list[str]:
    """Contrôle d'intégrité fourni par le validateur. Renvoie les raisons de refus."""
    from validator.handoff import verify_handoff

    check = verify_handoff(str(report), project_dir)
    return [] if check.ok else list(check.reasons)


def _override(state: PipelineState) -> tuple[list[str], str | None]:
    """Bloquants du v1 levés par un humain : ceux du dernier rapport, quand il vient de trancher pour le v2.
    Gardés ensuite pour toute la ligne (une revalidation après réparation ne les remet pas)."""
    waived = list(state.get("waived_codes") or [])
    last = (state.get("decisions") or [{}])[-1]
    if last.get("gate") == "validator_override" and last.get("decision") == "approve" and state.get("report_path"):
        try:
            findings = json.loads(Path(state["report_path"]).read_text(encoding="utf-8")).get("findings", [])
        except (OSError, ValueError):
            findings = []
        waived += [f["code"] for f in findings if f.get("severity") == "BLOCKER" and f["code"] not in waived]
        return waived, last.get("by")
    return waived, None


def validate(state: PipelineState) -> dict:
    run_dir = Path(state["run_dir"])
    # Après une réparation, on valide la copie corrigée (le projet d'origine n'est jamais modifié).
    project_dir = state.get("repaired_dir") or state["project_dir"]
    repairs = state.get("repair_count", 0)
    waived, by = _override(state)
    name = "validator" if not repairs else f"validator_after_repair_{repairs}"
    out = run_dir / (f"{name}_override" if waived else name)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "compatibility_report.json"
    started = time.monotonic()

    proc = run_validator(project_dir, report, run_dir / "snapshot", waived, by)

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
        refused = verify(report, project_dir)
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
    if waived:
        result.warnings.insert(0, f"bloquants du v1 levés par un humain (le v2 ne les confirme pas) : "
                                  f"{', '.join(waived)}")
    return {
        "report_path": str(report),
        "waived_codes": waived,
        # les agents lisent la copie figée ; à défaut (pas de snapshot), le projet d'origine
        "snapshot_dir": data.get("snapshot_path") or project_dir,
        "handoff_status": handoff.status.value,
        "artifacts": result.outputs,
        **_record("validator", result, started),
    }


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def make_agent_node(node_name: str, agent_fn: Callable[[AgentInput], AgentResult],
                    workdir_for: Callable[[PipelineState], Path] | None = None):
    """Enveloppe commune à tous les agents : prépare l'entrée, mesure le temps,
    transforme toute exception en FAILED, vérifie que la réponse respecte le schéma."""

    def node(state: PipelineState) -> dict:
        workdir = workdir_for(state) if workdir_for else Path(state["run_dir"]) / node_name
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


# ---------------------------------------------------------------------------
# Réparation : validateur -> réparation -> juge -> revalidation
# ---------------------------------------------------------------------------


def make_repair_node(agent_fn: Callable[[AgentInput], AgentResult]):
    """L'agent de réparation tourne dans runs/<run_id>/repair/attempt_<n>/.

    L'orchestrateur ne fait pas confiance aveuglément à la réponse : la copie réparée doit
    exister, être dans le workdir de la tentative, et le juge doit l'avoir approuvée.
    Sinon la réparation est refusée et le projet reste BLOCKED.
    """

    def workdir(state: PipelineState) -> Path:
        return Path(state["run_dir"]) / "repair" / f"attempt_{state.get('repair_count', 0) + 1}"

    run_agent = make_agent_node("repair", agent_fn, workdir)

    def node(state: PipelineState) -> dict:
        update = run_agent(state)
        attempt = state.get("repair_count", 0) + 1
        update["repair_count"] = attempt
        result = update["results"]["repair"]
        if result["can_proceed"]:
            problem = _check_repair(result["outputs"], workdir(state))
            if problem:
                result.update(status=AgentStatus.FAILED.value, can_proceed=False,
                              reasons=[*result["reasons"], f"réparation refusée par l'orchestrateur : {problem}"])
                update["history"][-1].update(result)
            else:
                update["repaired_dir"] = str(Path(result["outputs"]["repaired_project"]).resolve())
        return update

    node.__name__ = "repair"
    return node


def make_repair_review_gate():
    """Réparation acceptée par le juge mais sensible : un humain valide (Terminus).

    Même après l'accord humain, l'orchestrateur revérifie la copie (dans le workdir de la tentative,
    modifications déclarées) ; seul l'accord du juge est remplacé par celui de l'humain.
    """
    from .gates import make_gate

    gate = make_gate("repair_review", "repair_result", "gate_repair_review")

    def node(state: PipelineState) -> dict:
        update = gate(state)
        if update["decisions"][-1]["decision"] == "approve":
            outputs = state["results"]["repair"]["outputs"]
            workdir = Path(state["run_dir"]) / "repair" / f"attempt_{state.get('repair_count', 0)}"
            if _check_repair(outputs, workdir, human_approved=True) is None:
                update["repaired_dir"] = str(Path(outputs["repaired_project"]).resolve())
        return update

    node.__name__ = "gate_repair_review"
    return node


def _check_repair(outputs: dict, workdir: Path, human_approved: bool = False) -> str | None:
    if "repair_result" not in outputs or "repaired_project" not in outputs:
        return "sorties repair_result et repaired_project attendues"
    try:
        rr = RepairResult.model_validate_json(Path(outputs["repair_result"]).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - fichier illisible ou non conforme : réparation refusée
        return f"repair_result invalide ({type(exc).__name__}: {str(exc)[:200]})"
    project = Path(outputs["repaired_project"]).resolve()
    if not project.is_dir():
        return f"copie réparée introuvable : {project}"
    if not project.is_relative_to(workdir.resolve()):
        return "la copie réparée doit être écrite dans le workdir de la tentative"
    if not rr.judge.approved and not human_approved:
        return f"le juge n'a pas approuvé la réparation ({rr.judge.rationale[:200]})"
    if not rr.changes:
        return "aucune modification déclarée"
    return None


def _agent_name(node_name: str) -> str:
    return "agent1" if node_name == "propose_manifest" else node_name


def _record(node_name: str, result: AgentResult, started: float) -> dict:
    """Mise à jour de l'état : dernier résultat de l'étape + une ligne d'historique."""
    dumped = _dump(result, started)
    return {
        "results": {node_name: dumped},
        "history": [{"node": node_name, **dumped}],
        "last_step": node_name,
    }


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
