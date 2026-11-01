"""Handoff v2 : lignage, empreinte, copie figée, `can_proceed` selon le NOUVEAU statut, vérification côté consommateur.

Étend (sans les modifier) validator/handoff.py : l'empreinte, la copie figée et le modèle `Handoff` sont réutilisés.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from validator.contract import MODEL_ARTIFACT
from validator.fs import walk_project
from validator.handoff import INTERFACES, _feature_columns
from validator.handoff import _snapshot as _snapshot_copy
from validator.handoff import compute_project_hash
from validator.models import CarryForward, Handoff

from .schemas import (SCHEMA_VERSION, AdmissionStatus, CheckStatus, RunStatus, ValidatorReport, load_report, verify_seal)


def build_handoff(report: ValidatorReport, root: Path, data_summary: dict | None = None) -> Handoff:
    st, spec = report.status, report.project_spec
    carry = [CarryForward(code=c.check_id, message=c.note or c.name or c.check_id, source="finding")
             for c in report.checks if c.effective_status == CheckStatus.WARNING]
    carry += [CarryForward(code="LIMIT", message=lim, source="derived") for lim in report.limits[:5]]
    if st == AdmissionStatus.REJECTED:
        return Handoff(status="BLOCKED", can_proceed=False, requires_human_confirmation=False, reason="rejeté : " + "; ".join((report.policy.reasons if report.policy else [])[:3]))
    if st == AdmissionStatus.REPAIR_REQUIRED:
        return Handoff(status="BLOCKED", can_proceed=False, requires_human_confirmation=False,
                       reason="réparable : " + ", ".join((report.policy.blocking_checks if report.policy else [])[:5]))
    if st == AdmissionStatus.NEEDS_REVIEW:
        incomplete = report.run_status == RunStatus.FAILED_CLOSED
        return Handoff(status="INCOMPLETE" if incomplete else "AWAITING_CONFIRMATION", can_proceed=False,
                       requires_human_confirmation=not incomplete, reason="; ".join((report.policy.reasons if report.policy else [])[:3]) or "revue humaine requise")
    kw = {}
    if spec and spec.target and spec.data_path:
        kw = dict(task=spec.task, target=spec.target, primary_metric=spec.primary_metric, min_threshold=spec.min_threshold,
                  data_path=spec.data_path, feature_columns=_feature_columns(root, spec.data_path, spec.target))
    deps = {}
    return Handoff(status="READY_WITH_WARNINGS" if any(c.effective_status == CheckStatus.WARNING for c in report.checks) else "READY",
                   can_proceed=True, requires_human_confirmation=False, data_summary=data_summary or {},
                   interfaces=INTERFACES, model_artifact=MODEL_ARTIFACT, pinned_dependencies=deps,
                   observed_metrics={k: float(v) for k, v in (report.observed_metrics or {}).items()} or None,
                   carry_forward=carry, **kw)


def freeze_snapshot(root: Path, base, project_hash: str) -> str:
    files, _ = walk_project(root)
    return _snapshot_copy(SimpleNamespace(files=files, root=root), base, project_hash)


@dataclass
class ReportCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def verify_report(report, project_dir=None) -> ReportCheck:
    """Porte d'entrée de l'orchestrateur : version du schéma, sceau, et — pour un rapport du Validator — can_proceed et empreinte."""
    report = load_report(report)
    why = []
    if report.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        why.append(f"version de schéma non supportée : {report.schema_version}")
    if not verify_seal(report):
        why.append("rapport modifié après sa production (sceau invalide)")
    if report.report_type == "validator":
        if not report.can_proceed:
            why.append(f"passage de relais refusé : {report.status.value}")
        target = project_dir or report.snapshot_path
        if target:
            if not Path(target).is_dir():
                why.append(f"dossier introuvable : {target}")
            elif report.project_hash and compute_project_hash(target)[0] != report.project_hash:
                why.append("le contenu du projet a changé depuis la validation (empreinte différente)")
        elif report.can_proceed:
            why.append("aucun dossier à vérifier (ni project_dir, ni snapshot_path)")
    return ReportCheck(ok=not why, reasons=why)
