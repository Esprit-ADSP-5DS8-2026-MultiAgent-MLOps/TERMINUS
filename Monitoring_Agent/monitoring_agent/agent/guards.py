"""
Garde-fous déterministes : vérification des preuves et policy-as-code.

Le LLM propose. Ce fichier dispose. Rien ici n'appelle un LLM : les règles sont lisibles,
testées et versionnées.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from ..config import PolicyConfig
from .context import SIGNAL_KINDS_PERF, Case
from .schemas import ACTION_RISK, Diagnosis, JudgeVerdict, PolicyDecision, RuleCheck, Verification
from .schemas import EvidenceCheck as Check

# ---------------------------------------------------------------------------
# 1. Vérification des preuves (anti-hallucination)
# ---------------------------------------------------------------------------


# Cause annoncée <-> signaux calculés : une contradiction vérifiable sans LLM.
CAUSE_REQUIRES = {
    "concept_drift": ({"concept_drift_suspected"}, ("concept drift sans le signal concept_drift_suspected "
                      "(la perf estimée a baissé aussi : c'est plutôt un covariate shift)")),
    "covariate_shift": (SIGNAL_KINDS_PERF, "covariate shift sans aucune baisse de performance (c'est un benign_drift)"),
    "data_quality": ({"data_quality"}, "cause « qualité » sans signal de qualité des données"),
}
CAUSE_FORBIDS = {
    "benign_drift": (SIGNAL_KINDS_PERF, "« drift bénin » alors que la performance baisse"),
    "no_issue": (SIGNAL_KINDS_PERF | {"data_quality", "dataset_drift"}, "« aucun problème » alors que des signaux sont levés"),
}


CITATION = re.compile(r"INC-\d{4}")


def verify(diag: Diagnosis, facts: dict[str, float], features: list[str], rel_tol: float = 0.02,
           signals: set[str] | None = None, known_cases: set[str] | None = None) -> Verification:
    checks, problems = [], []
    for ev in diag.evidence:
        actual = facts.get(ev.fact)
        ok = actual is not None and abs(ev.value - actual) <= max(1e-3, rel_tol * abs(actual))
        checks.append(Check(fact=ev.fact, claimed=ev.value, actual=actual, ok=ok))
        if actual is None:
            problems.append(f"fait inexistant : {ev.fact}")
        elif not ok:
            problems.append(f"{ev.fact} : cité {ev.value:g}, réel {actual:g}")

    unknown = [f for f in diag.affected_features if f not in features]
    if unknown:
        problems.append(f"features inconnues : {unknown}")

    # Cohérence cause <-> preuves : une cause doit s'appuyer sur le bon type de fait.
    cited = {ev.fact for ev in diag.evidence}
    needs = {"data_quality": "quality.", "concept_drift": "perf.realized.", "covariate_shift": "drift."}
    prefix = needs.get(diag.root_cause)
    if prefix and not any(c.startswith((prefix, "tool.")) for c in cited):
        problems.append(f"cause « {diag.root_cause} » sans preuve de type {prefix}*")
    if diag.action != "no_action" and not diag.evidence:
        problems.append("action proposée sans aucune preuve")
    # RAG : un cas passé cité doit faire partie de ceux réellement fournis (citation vérifiable)
    cited_cases = set(CITATION.findall(f"{diag.summary} {diag.rationale}"))
    invented = sorted(cited_cases - (known_cases or set()))
    if invented:
        problems.append(f"cas cités mais non fournis par la mémoire : {invented}")

    if signals is not None:
        need = CAUSE_REQUIRES.get(diag.root_cause)
        if need and not signals & need[0]:
            problems.append(need[1])
        forbid = CAUSE_FORBIDS.get(diag.root_cause)
        if forbid and signals & forbid[0]:
            problems.append(forbid[1])
    return Verification(ok=not problems, checks=checks, problems=problems)


# Libellés : le LLM confond parfois « mesurée » (avec labels) et « estimée » (sans labels).
# La valeur est déjà vérifiée au-dessus ; ici on corrige seulement le texte affiché à l'humain.
MISLABELS = {
    "realized": (("sans label", "sans les label", "without label"), "performance mesurée avec les vrais labels"),
    "estimated": (("avec label", "avec les label", "avec les vrais label", "with label"), "performance estimée sans labels"),
}


def fix_claims(diag: Diagnosis) -> tuple[Diagnosis, list[str]]:
    """Remplace un libellé de preuve qui contredit le fait cité (réalisée <-> estimée)."""
    fixed, evidence = [], []
    for ev in diag.evidence:
        parts = ev.fact.split(".")
        kind = next((k for k in MISLABELS if k in parts), None)
        if kind and any(w in ev.claim.lower() for w in MISLABELS[kind][0]):
            fixed.append(ev.fact)
            ev = ev.model_copy(update={"claim": MISLABELS[kind][1]})
        evidence.append(ev)
    return (diag.model_copy(update={"evidence": evidence}) if fixed else diag), fixed


# ---------------------------------------------------------------------------
# 2. Policy : ce que l'agent a le droit de faire seul
# ---------------------------------------------------------------------------


def _parse_time(v) -> datetime | None:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None


def evaluate(diag: Diagnosis, case: Case, verification: Verification, verdict: JudgeVerdict | None,
             agreement: float, cfg: PolicyConfig | None = None, second_opinion: str | None = None) -> PolicyDecision:
    """`second_opinion` : action du décideur à règles (déterministe) sur le même cas.

    Principe d'autonomie : l'humain n'intervient que si c'est logiquement nécessaire.
    Quand un contrôle manque (juge indisponible, quota épuisé) ou hésite (juge « revise », confiance
    faible), deux méthodes indépendantes qui concluent la même chose (LLM + règles) suffisent pour agir.
    Restent humains : le haut risque (rollback), un rejet explicite du juge, des preuves invérifiables,
    un désaccord LLM / règles sur une action à risque, et le disjoncteur."""
    cfg = cfg or PolicyConfig()
    rules: list[RuleCheck] = []
    reasons: list[str] = []
    action = diag.action
    signals = case.signal_kinds()

    def rule(name: str, passed: bool, detail: str) -> bool:
        rules.append(RuleCheck(rule=name, passed=passed, detail=detail))
        return passed

    # -- a) Substitutions : l'action proposée est remplacée par une action plus sûre --------------
    if action == "request_retraining":
        if not rule("R1 réentraînement = baisse de performance",
                    bool(signals & SIGNAL_KINDS_PERF),
                    "un drift seul ne prouve pas une dégradation (spec : drift ≠ dégradation)"):
            action = "increase_monitoring"
            reasons.append("pas de baisse de performance (mesurée ou estimée) : on surveille au lieu de réentraîner")
        elif not rule("R2 pas de réentraînement sur des données cassées", not case.has_quality_problem(),
                      "réentraîner sur des données invalides, c'est apprendre le bug"):
            action = "notify_data_team"
            reasons.append("problème de qualité des données : corriger la source avant de réentraîner")
        elif not rule("R3 des labels récents existent",
                      any(w.performance and w.performance.labels_available
                          for w in case.history if w.dataset_drift.n_drifted > 0 or w is case.current),
                      "il faut des données labellisées de la nouvelle période pour réentraîner"):
            action = "open_incident"
            reasons.append("pas encore de labels sur la période dégradée : incident ouvert, réentraînement plus tard")
        else:
            last = [_parse_time(d.get("created_at")) for d in case.decisions if d.get("action") == "request_retraining"
                    and d.get("status") == "executed"]
            last = [t for t in last if t]
            recent = any(case.now - t < timedelta(hours=cfg.retrain_cooldown_h) for t in last)
            if not rule(f"R4 délai entre deux réentraînements ({cfg.retrain_cooldown_h} h)", not recent,
                        "évite une boucle de réentraînement"):
                action = "open_incident"
                reasons.append("une demande de réentraînement est déjà en cours")

    # -- b) Mode : seul ou avec un humain -----------------------------------------------------------
    risk = ACTION_RISK[action]
    human = []
    if risk == "high":
        human.append("action à haut risque (change le modèle servi)")
        rule("R5 haut risque = humain", False, "rollback / remplacement : jamais automatique")
    if not rule("R6 preuves vérifiées", verification.ok, "; ".join(verification.problems) or "toutes les preuves sont exactes"):
        human.append("des preuves citées sont fausses ou absentes")
    rules_agree = second_opinion is not None and second_opinion == action
    agree_txt = f"le décideur à règles conclut aussi « {second_opinion} »" if rules_agree else \
        f"le décideur à règles conclut « {second_opinion} »"
    if verdict is not None and verdict.verdict == "approve":
        rule("R7 avis du juge", True, f"approuvé : {verdict.critique[:160]}")
    elif verdict is not None and verdict.verdict == "reject":
        rule("R7 avis du juge", False, f"rejeté : {verdict.critique[:160]}")
        human.append("le juge rejette le diagnostic")
    else:
        why = "juge indisponible" if verdict is None else f"le juge demande encore une révision ({verdict.critique[:100]})"
        if not rule("R7 avis du juge (ou second avis)", risk == "low" or rules_agree, f"{why} ; {agree_txt}"):
            human.append(f"{why}, et le second avis déterministe ne concorde pas")
    confident = diag.confidence >= cfg.min_confidence and agreement >= cfg.min_agreement
    if not rule("R8 confiance et accord", confident or risk == "low" or rules_agree,
                f"confiance {diag.confidence:.2f} (min {cfg.min_confidence}), accord {agreement:.0%} "
                f"(min {cfg.min_agreement:.0%})" + ("" if confident else f" ; {agree_txt}")):
        human.append("diagnostic incertain et non confirmé par les règles")
    recent_auto = [d for d in case.decisions if d.get("mode") == "auto"
                   and (t := _parse_time(d.get("created_at"))) and case.now - t < timedelta(hours=24)]
    if not rule(f"R9 budget d'actions automatiques ({cfg.max_auto_actions_24h}/24 h)",
                len(recent_auto) < cfg.max_auto_actions_24h, f"{len(recent_auto)} actions automatiques sur 24 h"):
        human.append("budget d'actions automatiques épuisé (disjoncteur)")

    mode = "human" if human else "auto"
    return PolicyDecision(proposed_action=diag.action, final_action=action, mode=mode, rules=rules,
                          reasons=reasons + human)
