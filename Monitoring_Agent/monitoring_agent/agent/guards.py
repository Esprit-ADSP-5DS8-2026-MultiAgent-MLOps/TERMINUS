"""
Garde-fous déterministes : vérification des preuves et policy-as-code.

Le LLM PROPOSE. Ce fichier DISPOSE. Rien ici n'appelle un LLM : les règles sont lisibles,
testées, versionnées, et la prof peut les lire en 5 minutes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..config import PolicyConfig
from .context import SIGNAL_KINDS_PERF, Case
from .schemas import ACTION_RISK, Diagnosis, JudgeVerdict, PolicyDecision, RuleCheck, Verification
from .schemas import EvidenceCheck as Check

# ---------------------------------------------------------------------------
# 1. Vérification des preuves (anti-hallucination)
# ---------------------------------------------------------------------------


def verify(diag: Diagnosis, facts: dict[str, float], features: list[str], rel_tol: float = 0.02) -> Verification:
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
    return Verification(ok=not problems, checks=checks, problems=problems)


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
             agreement: float, cfg: PolicyConfig | None = None) -> PolicyDecision:
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
    judge_ok = verdict is not None and verdict.verdict == "approve"
    if verdict is None:
        rule("R7 avis du juge", risk == "low", "juge indisponible : accepté seulement pour une action à faible risque")
        if risk != "low":
            human.append("juge indisponible pour une action à risque")
    elif not rule("R7 avis du juge", judge_ok, f"verdict « {verdict.verdict} » : {verdict.critique[:160]}"):
        human.append(f"le juge n'approuve pas ({verdict.verdict})")
    confident = diag.confidence >= cfg.min_confidence and agreement >= cfg.min_agreement
    if not rule("R8 confiance et accord", confident or risk == "low",
                f"confiance {diag.confidence:.2f} (min {cfg.min_confidence}), accord {agreement:.0%} (min {cfg.min_agreement:.0%})"):
        human.append("diagnostic incertain")
    recent_auto = [d for d in case.decisions if d.get("mode") == "auto"
                   and (t := _parse_time(d.get("created_at"))) and case.now - t < timedelta(hours=24)]
    if not rule(f"R9 budget d'actions automatiques ({cfg.max_auto_actions_24h}/24 h)",
                len(recent_auto) < cfg.max_auto_actions_24h, f"{len(recent_auto)} actions automatiques sur 24 h"):
        human.append("budget d'actions automatiques épuisé (disjoncteur)")

    mode = "human" if human else "auto"
    return PolicyDecision(proposed_action=diag.action, final_action=action, mode=mode, rules=rules,
                          reasons=reasons + human)
