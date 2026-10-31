"""
Le superviseur : après chaque étape, il choisit la suite du pipeline.

1. Les RÈGLES calculent les actions autorisées à ce moment-là (routing.py).
2. Le DÉCIDEUR (règles ou Gemini) en choisit une et l'explique.
3. Le superviseur VÉRIFIE que l'action est autorisée ; sinon, ou si le LLM
   ne répond pas, il applique la règle (solution de secours).
4. LangGraph envoie le run vers l'étape choisie (Command(goto=...)).

Garanties, quel que soit le décideur :
- une validation humaine ne peut jamais être sautée (elle n'est jamais remplacée
  par l'étape qui la suit dans les actions autorisées) ;
- un projet refusé ou un quality gate raté ne mène jamais au déploiement ;
- une étape en échec est relancée au plus MAX_STEP_RETRIES fois sans humain.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from common.schemas import utcnow

from . import routing
from .llm import get_decider
from .state import PipelineState

MAX_STEP_RETRIES = 2

# dernière étape terminée -> règle qui donne la suite
ROUTERS = {
    "validator": routing.after_validate,
    "propose_manifest": routing.after_propose_manifest,
    "repair": routing.after_repair,
    "agent1": routing.after_agent1,
    "agent2": routing.after_agent2,
    "agent3": routing.after_agent3,
    "agent4": routing.after_agent4,
    "gate_manifest": routing.after_gate_manifest,
    "gate_champion": routing.after_gate_champion,
    "gate_retraining": routing.after_gate_retraining,
    "gate_failure": routing.after_gate_failure,
}

RETRYABLE = {"validator", "propose_manifest", "agent1", "agent2", "agent3", "agent4"}
STEP_NODE = {"validator": "validate"}  # nom du résultat -> nom du nœud à relancer

LABELS = {
    "validator": "Validateur",
    "propose_manifest": "Agent 1 (proposition du manifeste)",
    "repair": "Réparation automatique (validateur v2)",
    "agent1": "Agent 1 (analyse)",
    "agent2": "Agent 2 (MLflow)",
    "agent3": "Agent 3 (déploiement)",
    "agent4": "Agent 4 (monitoring)",
    "gate_manifest": "validation humaine du manifeste",
    "gate_champion": "validation humaine de la mise en production",
    "gate_retraining": "validation humaine du réentraînement",
    "gate_failure": "décision humaine après un échec",
}

DESCRIPTIONS = {
    "agent1": "Lancer l'Agent 1 : analyser le projet et ses données",
    "agent2": "Lancer l'Agent 2 : entraîner et suivre le modèle dans MLflow, appliquer le quality gate",
    "agent3": "Lancer l'Agent 3 : construire, tester et déployer l'API du modèle",
    "agent4": "Lancer l'Agent 4 : surveiller le modèle déployé",
    "propose_manifest": "Demander à l'Agent 1 de proposer un mlops.yaml",
    "repair": "Tenter une réparation automatique du projet (sur une copie, jugée, puis revalidée)",
    "validate": "Revalider la copie réparée du projet",
    "gate_manifest": "Demander à un humain de valider le manifeste proposé",
    "gate_champion": "Demander à un humain de valider la mise en production du modèle",
    "gate_retraining": "Demander à un humain s'il faut réentraîner le modèle",
    "gate_failure": "Demander à un humain s'il faut relancer l'étape qui a échoué",
    "end_done": "Terminer : le modèle est déployé et surveillé",
    "end_blocked": "Arrêter : le projet ne respecte pas le contrat",
    "end_failed": "Arrêter le run en échec",
    "end_rejected": "Arrêter : refusé par un humain",
    "end_manifest_proposed": "Terminer : le manifeste est validé, le projet doit être relancé avec",
    "end_champion_kept": "Terminer : le nouveau modèle ne bat pas le champion, qui reste en production",
}


def options_for(state: PipelineState) -> tuple[str, dict[str, str]]:
    """La décision des règles et toutes les actions autorisées (identifiant -> description)."""
    last = state["last_step"]
    rule = ROUTERS[last](state)

    if last == "gate_failure":
        if rule == "end_failed":
            return rule, {rule: DESCRIPTIONS[rule]}
        step = state["failed_step"]
        return "retry", {"retry": f"Relancer l'étape {LABELS[step]}, comme l'humain l'a décidé"}

    options = {rule: DESCRIPTIONS[rule]}
    if rule == "end_failed" and last in RETRYABLE:
        done = state.get("attempts", {}).get(last, 0)
        if done < MAX_STEP_RETRIES:
            options["retry"] = (f"Relancer l'étape {LABELS[last]} "
                                f"(nouvelle tentative {done + 1} sur {MAX_STEP_RETRIES})")
        options["gate_failure"] = DESCRIPTIONS["gate_failure"]
    # Après un refus du validateur, pas d'alternative à la réparation : c'est l'agent de réparation
    # (et son juge) qui sait ce qu'il peut corriger, pas le superviseur.
    if last == "agent4" and rule == "gate_retraining":
        options["end_done"] = "Ne pas demander de réentraînement : continuer la surveillance et terminer"
    return rule, options


def context_for(state: PipelineState) -> dict:
    """Ce que le décideur voit du run : court, sans secret, sans contenu de fichier."""
    last = state["last_step"]
    ctx: dict = {
        "projet": Path(state.get("project_dir", "?")).name,
        "etape_terminee": LABELS.get(last, last),
        "etapes_deja_executees": [LABELS.get(h["node"], h["node"]) for h in state.get("history", [])],
        "reentrainements_deja_faits": state.get("retrain_count", 0),
        "reparations_deja_faites": state.get("repair_count", 0),
    }
    result = state.get("results", {}).get(last)
    if result:
        ctx["resultat"] = {
            "statut": result["status"],
            "raisons": [r[:300] for r in result.get("reasons", [])[:3]],
            "avertissements": [w[:300] for w in result.get("warnings", [])[:3] if not w.startswith("stub")],
        }
        ctx["tentatives_deja_faites_pour_cette_etape"] = state.get("attempts", {}).get(last, 0)
    if last.startswith("gate_") and state.get("decisions"):
        d = state["decisions"][-1]
        ctx["decision_humaine"] = {"decision": d["decision"], "commentaire": d.get("comment", "")}
    return ctx


def supervisor(state: PipelineState) -> Command:
    last = state["last_step"]
    rule, options = options_for(state)
    mode = state.get("supervisor_mode", "rules")

    context = context_for(state)
    action, explanation, fallback = rule, options[rule], None
    try:
        decision = get_decider(mode).decide(context, options, rule)
        if decision.action.strip() in options:
            action = decision.action.strip()
            explanation = decision.explanation.strip()[:500] or options[action]
        else:
            fallback = f"action « {decision.action[:80]} » non autorisée : règle appliquée"
    except Exception as exc:  # LLM injoignable, clé absente, réponse illisible...
        fallback = f"{type(exc).__name__}: {exc}"[:300] + " : règle appliquée"

    # Chaque décision est enregistrée avec ce que le décideur a vu (context, options) :
    # on peut ainsi relire, après coup, pourquoi chaque décision a été prise.
    update: dict = {
        "supervision": [{
            "after": last,
            "action": action,
            "explanation": explanation,
            "decider": mode,
            "fallback": fallback,
            "rule_choice": rule,
            "context": context,
            "options": dict(options),  # identifiant -> description, comme dans la question posée
            "at": utcnow().isoformat(),
        }],
    }
    goto = action
    if action == "retry":
        step = state["failed_step"] if last == "gate_failure" else last
        goto = STEP_NODE.get(step, step)
        update["attempts"] = {step: state.get("attempts", {}).get(step, 0) + 1}
    elif action == "gate_failure":
        update["failed_step"] = last
    return Command(goto=goto, update=update)


DESTINATIONS = tuple(sorted({d for d in DESCRIPTIONS} | {"validate"}))
