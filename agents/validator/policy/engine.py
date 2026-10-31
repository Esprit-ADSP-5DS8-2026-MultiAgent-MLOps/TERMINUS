"""Moteur de politique : agrège contrôles + problèmes et CALCULE le statut. Petit, déterministe, sans LLM.

Règles (toutes issues du contrat, aucune règle de projet dans ce fichier) :
  1. contrôle OBLIGATOIRE en FAIL                      -> bloque : REJECTED / REPAIR_REQUIRED / NEEDS_REVIEW selon `on_fail` du contrat ;
  2. contrôle OBLIGATOIRE NOT_CHECKED (ou non rapporté),
     ou NOT_APPLICABLE sans justification valable      -> empêche ACCEPTED (donc NEEDS_REVIEW) : un LLM ne peut pas « sauter » un contrôle ;
  3. contrôle RECOMMANDÉ en FAIL                       -> WARNING (n'empêche pas ACCEPTED) ;
  4. problème CRITICAL sans contrôle bloquant          -> bloque (REJECTED si catégorie security) ;
  5. exécution exigée par le contrat mais non réussie   -> NEEDS_REVIEW au mieux ;
  6. priorité : REJECTED > REPAIR_REQUIRED > NEEDS_REVIEW > ACCEPTED.
"""
from __future__ import annotations

from agentic_core.schemas import (ADMISSION_RANK, AdmissionStatus, Check, CheckStatus, Issue, PolicyOutcome, Severity)

from .model import AdmissionContract


def _worst(a: Check, b: Check) -> Check:
    order = {CheckStatus.FAIL: 4, CheckStatus.WARNING: 3, CheckStatus.NOT_CHECKED: 2, CheckStatus.NOT_APPLICABLE: 1, CheckStatus.PASS: 0}
    return a if order[a.status] >= order[b.status] else b


def evaluate(contract: AdmissionContract, checks: list[Check], issues: list[Issue], *, executed_ok: bool | None = None
             ) -> tuple[PolicyOutcome, list[Check]]:
    reported: dict[str, Check] = {}
    extras: list[Check] = []
    for c in checks:
        if contract.check(c.check_id) is None:
            extras.append(c)
        else:
            reported[c.check_id] = _worst(reported[c.check_id], c) if c.check_id in reported else c

    reasons: list[str] = []
    blocking: dict[AdmissionStatus, list[str]] = {s: [] for s in AdmissionStatus}
    review: list[str] = []
    warnings: list[str] = []
    out: list[Check] = []

    for spec in contract.checks:
        c = reported.get(spec.id)
        mandatory = spec.requirement == "mandatory"
        if c is None:
            c = Check(check_id=spec.id, status=CheckStatus.NOT_CHECKED, note="contrôle non rapporté par l'analyse")
        c = c.model_copy(update={"name": c.name or spec.title, "level": spec.level, "mandatory": mandatory})
        eff = c.status
        if eff == CheckStatus.NOT_APPLICABLE:
            valid = any(n.id == c.na_reason for n in spec.not_applicable_when) and bool(c.evidence)
            if not valid:
                eff = CheckStatus.NOT_CHECKED
                reasons.append(f"{spec.id} : NOT_APPLICABLE sans justification valable (raison « {c.na_reason} », preuve {'présente' if c.evidence else 'absente'})")
        if eff == CheckStatus.FAIL:
            if mandatory:
                blocking[AdmissionStatus(spec.on_fail)].append(spec.id)
                reasons.append(f"{spec.id} obligatoire en FAIL → {spec.on_fail}")
            else:
                eff = CheckStatus.WARNING
                warnings.append(spec.id)
        elif eff == CheckStatus.WARNING:
            warnings.append(spec.id)
        elif eff == CheckStatus.NOT_CHECKED:
            if mandatory:
                review.append(spec.id)
                reasons.append(f"{spec.id} obligatoire mais non vérifié → ACCEPTED impossible")
            else:
                warnings.append(spec.id)
        out.append(c.model_copy(update={"effective_status": eff}))
    out += [c.model_copy(update={"mandatory": False, "effective_status": c.status}) for c in extras]

    covered = {i for ids in blocking.values() for i in ids}
    for i in issues:
        if i.severity == Severity.CRITICAL and i.check_id not in covered:
            s = AdmissionStatus.REJECTED if i.category == "security" else AdmissionStatus.REPAIR_REQUIRED
            blocking[s].append(i.check_id)
            reasons.append(f"problème CRITICAL {i.issue_id} ({i.check_id}) → {s.value}")

    if contract.execution_required and executed_ok is not True:
        reasons.append("exécution exigée par le contrat mais non réussie → ACCEPTED impossible")
        review.append("EXECUTION")

    if blocking[AdmissionStatus.REJECTED]:
        status = AdmissionStatus.REJECTED
    elif blocking[AdmissionStatus.REPAIR_REQUIRED]:
        status = AdmissionStatus.REPAIR_REQUIRED
    elif blocking[AdmissionStatus.NEEDS_REVIEW] or review:
        status = AdmissionStatus.NEEDS_REVIEW
    else:
        status = AdmissionStatus.ACCEPTED
    blocking_ids = sorted({i for ids in blocking.values() for i in ids})
    return PolicyOutcome(contract_id=contract.id, contract_version=contract.version, status=status,
                         can_proceed=status == AdmissionStatus.ACCEPTED, reasons=reasons or ["tous les contrôles obligatoires passent"],
                         blocking_checks=blocking_ids, review_checks=sorted(set(review)), warning_checks=sorted(set(warnings))), out


def worst_status(a: AdmissionStatus, b: AdmissionStatus) -> AdmissionStatus:
    return a if ADMISSION_RANK[a] >= ADMISSION_RANK[b] else b
