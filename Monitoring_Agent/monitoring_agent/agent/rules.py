"""
Décideur SANS LLM, à base de règles.

Deux usages :
- repli quand aucun LLM n'est configuré ou joignable (l'agent continue de fonctionner) ;
- BASELINE de l'évaluation (spec §10) : on compare règles / LLM de base / LLM fine-tuné.
"""

from __future__ import annotations

from .context import Case, case_facts
from .schemas import Diagnosis, Evidence


def diagnose_by_rules(case: Case) -> Diagnosis:
    w, facts, m = case.current, case_facts(case), case.mapping.primary_metric
    kinds = case.signal_kinds()
    drifted = w.dataset_drift.drifted_features

    def ev(*keys: str) -> list[Evidence]:
        return [Evidence(fact=k, value=facts[k], claim=k) for k in keys if k in facts]

    top_psi = sorted((k for k in facts if k.startswith("drift.") and k.endswith(".psi")),
                     key=lambda k: -facts[k])[:3]

    if case.has_quality_problem():
        q = [k for k in facts if k.startswith("quality.")][:3]
        return Diagnosis(root_cause="data_quality", summary="Problème de qualité des données détecté.",
                         affected_features=sorted({k.split(".")[1] for k in q}), evidence=ev(*q),
                         action="notify_data_team", rationale="La source de données doit être corrigée.", confidence=0.8)
    if "concept_drift_suspected" in kinds:
        return Diagnosis(root_cause="concept_drift", summary="Performance mesurée en baisse, estimation stable.",
                         affected_features=drifted, evidence=ev(f"perf.realized.{m}", f"perf.estimated.{m}", "perf.bound"),
                         action="request_retraining", rationale="La relation features -> cible a changé.", confidence=0.7)
    if kinds & {"performance_drop_realized", "performance_drop_estimated"}:
        key = f"perf.realized.{m}" if "performance_drop_realized" in kinds else f"perf.estimated.{m}"
        return Diagnosis(root_cause="covariate_shift" if drifted else "concept_drift",
                         summary="Baisse de performance associée au drift." if drifted else "Baisse de performance.",
                         affected_features=drifted, evidence=ev(key, "perf.bound", *top_psi),
                         action="request_retraining", rationale="Performance sous la variabilité normale.", confidence=0.7)
    if drifted or "prediction_drift" in kinds:
        return Diagnosis(root_cause="benign_drift", summary="Drift sans baisse de performance.",
                         affected_features=drifted, evidence=ev(*top_psi),
                         action="increase_monitoring", rationale="Drift ≠ dégradation : on surveille.", confidence=0.7)
    return Diagnosis(root_cause="no_issue", summary="Rien d'anormal.", action="no_action",
                     rationale="Aucun signal significatif.", confidence=0.9)
