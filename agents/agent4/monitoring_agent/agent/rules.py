"""
Décideur sans LLM, à base de règles.

Deux usages :
- repli quand aucun LLM n'est configuré ou joignable (l'agent continue de fonctionner) ;
- baseline de l'évaluation (spec §10) : on compare règles / LLM de base / LLM fine-tuné.
"""

from __future__ import annotations

from .context import Case, case_facts
from .schemas import Diagnosis, Evidence

QUALITY_TXT = {"missing_values": "taux de valeurs manquantes", "unseen_category": "part de catégories inconnues",
               "out_of_range": "part de valeurs hors de l'intervalle d'entraînement", "type_mismatch": "part de types invalides",
               "missing_column": "colonne absente"}


def claim_for(key: str) -> str:
    """Ce que montre un fait, en clair (les règles n'écrivent pas comme un LLM, mais doivent rester lisibles)."""
    parts = key.split(".")
    if key.startswith("perf.realized."):
        return "performance mesurée avec les vrais labels"
    if key.startswith("perf.estimated."):
        return "performance estimée sans labels"
    if key == "perf.bound":
        return "limite de la variabilité normale (référence)"
    if key.startswith("drift.") and key.endswith(".psi"):
        return f"PSI de {parts[1]} (0,25 = drift majeur)"
    if key.startswith("quality."):
        return f"{QUALITY_TXT.get(parts[-1], parts[-1])} sur {parts[1]}"
    return key


def procedure(case: Case) -> tuple[str, str]:
    """La procédure du prompt (prompts.PROCEDURE), en code : (cause, action).

    Source unique de vérité : le décideur à règles, l'oracle du dataset de fine-tuning et le prompt
    suivent le même ordre. Un changement ici doit être répercuté dans prompts.PROCEDURE."""
    kinds = case.signal_kinds()
    perf = case.current.performance
    labels = bool(perf and perf.labels_available)
    if case.has_quality_problem():
        return "data_quality", "notify_data_team"
    if kinds & {"performance_drop_realized", "performance_drop_estimated"}:
        cause = "concept_drift" if "concept_drift_suspected" in kinds else "covariate_shift"
        return cause, "request_retraining" if labels else "open_incident"
    if case.current.dataset_drift.n_drifted or "prediction_drift" in kinds:
        return "benign_drift", "increase_monitoring"
    return "no_issue", "no_action"


def diagnose_by_rules(case: Case) -> Diagnosis:
    w, facts, m = case.current, case_facts(case), case.mapping.primary_metric
    drifted = w.dataset_drift.drifted_features
    cause, action = procedure(case)

    def ev(*keys: str) -> list[Evidence]:
        return [Evidence(fact=k, value=facts[k], claim=claim_for(k)) for k in keys if k in facts]

    top_psi = sorted((k for k in facts if k.startswith("drift.") and k.endswith(".psi")),
                     key=lambda k: -facts[k])[:3]
    perf_key = f"perf.realized.{m}" if f"perf.realized.{m}" in facts else f"perf.estimated.{m}"
    labels = "" if action != "open_incident" else " Pas encore de labels : incident ouvert, réentraînement plus tard."

    if cause == "data_quality":
        q = sorted((k for k in facts if k.startswith("quality.")), key=lambda k: -facts[k])[:3]
        return Diagnosis(root_cause=cause, summary="Problème de qualité des données détecté.",
                         affected_features=sorted({k.split(".")[1] for k in q}), evidence=ev(*q),
                         action=action, rationale="La source de données doit être corrigée avant tout réentraînement.",
                         confidence=0.8)
    if cause == "concept_drift":
        return Diagnosis(root_cause=cause, summary="Performance mesurée en baisse, estimation stable.",
                         affected_features=drifted, evidence=ev(f"perf.realized.{m}", f"perf.estimated.{m}", "perf.bound"),
                         action=action, rationale="La relation features -> cible a changé." + labels, confidence=0.7)
    if cause == "covariate_shift":
        return Diagnosis(root_cause=cause, summary="Baisse de performance associée au drift des entrées.",
                         affected_features=drifted, evidence=ev(perf_key, "perf.bound", *top_psi),
                         action=action, rationale="Performance sous la variabilité normale." + labels, confidence=0.7)
    if cause == "benign_drift":
        return Diagnosis(root_cause=cause, summary="Drift sans baisse de performance.",
                         affected_features=drifted, evidence=ev(*top_psi),
                         action=action, rationale="Drift ≠ dégradation : on surveille.", confidence=0.7)
    return Diagnosis(root_cause="no_issue", summary="Rien d'anormal.", action="no_action",
                     rationale="Aucun signal significatif.", confidence=0.9)
