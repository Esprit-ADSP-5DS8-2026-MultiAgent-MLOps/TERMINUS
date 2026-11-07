"""
Décisions de routage : après chaque étape, quelle est la suivante ?

Ces fonctions ne modifient rien : elles lisent l'état et renvoient le nom du nœud suivant.
"""

from __future__ import annotations

from pathlib import Path

from .gates import MAX_RETRAINS, last_decision
from .state import PipelineState

MAX_REPAIRS = 2  # tentatives de réparation automatique avant d'abandonner (projet BLOCKED)
REPAIRABLE = {"BLOCKED", "INCOMPLETE"}


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
    if state.get("waived_codes") and _only_unwaivable(state):
        return "end_blocked"   # un humain a déjà tranché ; ce qui reste ne se lève pas (sécurité)
    if state.get("handoff_status") in REPAIRABLE and state.get("repair_count", 0) < MAX_REPAIRS:
        return "repair"
    return "end_blocked"


def _only_unwaivable(state: PipelineState) -> bool:
    """Après un désaccord tranché : les bloquants restants sont-ils tous de ceux qu'un humain ne lève pas ?"""
    import json

    from validator.contract import NEVER_WAIVED

    try:
        findings = json.loads(Path(state["report_path"]).read_text(encoding="utf-8")).get("findings", [])
    except (KeyError, OSError, ValueError):
        return False
    codes = {f["code"] for f in findings if f.get("severity") == "BLOCKER"}
    return bool(codes) and codes <= NEVER_WAIVED


def after_repair(state: PipelineState) -> str:
    # Réparation jugée et acceptée : on revalide la copie corrigée ; sinon le projet reste refusé.
    r = _result(state, "repair")
    if r["status"] == "NEEDS_APPROVAL":   # un humain tranche (Terminus) : réparation sensible ou désaccord v1/v2
        return "gate_repair_review" if "repair_result" in r["outputs"] else "gate_validator_override"
    return "validate" if r["can_proceed"] else "end_blocked"


def after_gate_repair_review(state: PipelineState) -> str:
    # Approuvée et vérifiée (copie dans le workdir de la tentative) : revalidation ; sinon arrêt.
    approved = last_decision(state) == "approve"
    copy = _result(state, "repair")["outputs"].get("repaired_project")
    return ("validate" if approved and copy and state.get("repaired_dir") == str(Path(copy).resolve())
            else "end_blocked")


def after_gate_validator_override(state: PipelineState) -> str:
    # L'humain donne raison au v2 : on revalide avec les bloquants du v1 levés (rapportés en avertissement).
    # Les agents reçoivent ainsi un rapport complet (tâche, cible, métriques observées) qui trace la décision,
    # au lieu d'un rapport BLOCKED sans manifeste.
    return "validate" if last_decision(state) == "approve" else "end_blocked"


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
    if r["status"] == "WARNING" and not r["can_proceed"]:
        return "end_champion_kept"  # le candidat ne bat pas le champion : il reste en production
    if state.get("retrain_count", 0) and not r["can_proceed"] and "mlflow_metadata" in r["outputs"]:
        # réentraînement : le candidat rate le quality gate ; le modèle déployé continue de servir
        return "end_champion_kept"
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


def after_gate_failure(state: PipelineState) -> str:
    # L'humain a accepté de relancer l'étape en échec, ou a arrêté le run.
    return state["failed_step"] if last_decision(state) == "approve" else "end_failed"
