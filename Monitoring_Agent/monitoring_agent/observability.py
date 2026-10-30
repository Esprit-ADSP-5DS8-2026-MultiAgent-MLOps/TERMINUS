"""
LLMOps avec Langfuse (v5, OpenTelemetry) - totalement optionnel : sans clés, tout est no-op.

Ce qu'on envoie, et pourquoi :
- une trace par décision (« agent4.decision »), avec un span par nœud du graphe (retriever, tool, evaluator,
  guardrail…) et une génération par appel LLM (modèle, prompt, réponse, jetons, latence) : on rejoue et on
  compare chaque décision, et on voit où passent le temps et les jetons ;
- des scores sur la trace : preuves vérifiées, notes du juge, accord du vote, autonomie, action finale,
  et la décision humaine quand il y en a une (feedback humain = meilleure étiquette pour améliorer l'agent) ;
- les prompts versionnés (prompt management) : `python -m monitoring_agent langfuse push-prompts` ;
- un dataset + des expériences : les cas de test du fine-tuning, rejoués par n'importe quel modèle.

Les clés (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL) sont lues dans Monitoring_Agent/.env.
"""

from __future__ import annotations

import contextlib
import logging
from functools import cache
from typing import Any

log = logging.getLogger("monitoring_agent.observability")

NODE_TYPES = {"prepare": "retriever", "investigate": "agent", "diagnose": "chain", "verify": "guardrail",
              "judge": "evaluator", "policy": "guardrail", "await_human": "span", "act": "tool"}


@cache
def client():
    """Client Langfuse, ou None si les clés manquent (tests, machine de l'équipe sans compte)."""
    from .agent.llm import _settings
    s = _settings()
    if not (s.get("LANGFUSE_PUBLIC_KEY") and s.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(public_key=s["LANGFUSE_PUBLIC_KEY"], secret_key=s["LANGFUSE_SECRET_KEY"],
                        base_url=s.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                        environment=s.get("LANGFUSE_ENVIRONMENT", "dev"), release="agent4")
    except Exception:
        log.warning("Langfuse indisponible", exc_info=True)
        return None


def enabled() -> bool:
    return client() is not None


@contextlib.contextmanager
def observation(name: str, as_type: str = "span", **kw):
    """Span / génération Langfuse si actif, sinon rien. Renvoie l'observation (ou None)."""
    lf = client()
    if lf is None:
        yield None
        return
    try:
        cm = lf.start_as_current_observation(name=name, as_type=as_type, **kw)
        obs = cm.__enter__()
    except Exception:
        log.debug("observation Langfuse impossible", exc_info=True)
        yield None
        return
    try:
        yield obs
    except BaseException as exc:
        with contextlib.suppress(Exception):
            obs.update(level="ERROR", status_message=str(exc)[:300])
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


@contextlib.contextmanager
def decision_trace(run_id: str, project: str, metadata: dict):
    """Trace racine d'une décision ; renvoie l'id de trace (pour les scores) ou None."""
    lf = client()
    if lf is None:
        yield None
        return
    from langfuse import propagate_attributes
    with propagate_attributes(session_id=project, trace_name="agent4.decision", tags=["agent4", project],
                              metadata={k: str(v) for k, v in metadata.items()}), \
            observation("agent4.decision", as_type="agent", input={"run_id": run_id, **metadata}) as root:
        yield lf.get_current_trace_id() if root is not None else None


def finish_decision(run) -> None:
    """Sortie de la trace racine : sans ça, Langfuse affiche « undefined » dans Output."""
    lf = client()
    if lf is None:
        return
    p, d = run.policy, run.diagnosis
    output = {"status": run.status, "final_action": p.final_action if p else None, "mode": p.mode if p else None,
              "root_cause": d.root_cause if d else None, "summary": d.summary if d else None,
              "outputs": run.outputs, "error": run.error}
    with contextlib.suppress(Exception):
        lf.update_current_span(output=output)


def update(obs, **kw) -> None:
    if obs is not None:
        with contextlib.suppress(Exception):
            obs.update(**kw)


def score_run(trace_id: str | None, run) -> None:
    """Scores automatiques d'une décision (les mêmes que l'évaluation hors ligne)."""
    lf = client()
    if lf is None or trace_id is None:
        return
    scores: list[tuple[str, Any, str]] = []
    if run.verification:
        scores.append(("evidence_verified", float(run.verification.ok), "BOOLEAN"))
    if run.verdicts:
        v = run.verdicts[-1]
        scores.append(("judge_verdict", v.verdict, "CATEGORICAL"))
        scores += [(f"judge_{k}", float(s), "NUMERIC") for k, s in v.scores.items()]
    scores.append(("self_consistency", float(run.agreement), "NUMERIC"))
    if run.policy:
        scores.append(("autonomous", float(run.policy.mode == "auto"), "BOOLEAN"))
        scores.append(("final_action", run.policy.final_action, "CATEGORICAL"))
    for name, value, kind in scores:
        with contextlib.suppress(Exception):
            lf.create_score(name=name, value=value, trace_id=trace_id, data_type=kind)


def score_human(trace_id: str | None, decision: str, comment: str = "") -> None:
    lf = client()
    if lf is None or trace_id is None:
        return
    with contextlib.suppress(Exception):
        lf.create_score(name="human_feedback", value=decision, trace_id=trace_id, data_type="CATEGORICAL",
                        comment=comment or None)


def trace_url(trace_id: str | None) -> str | None:
    lf = client()
    if lf is None or trace_id is None:
        return None
    try:
        return lf.get_trace_url(trace_id=trace_id)
    except Exception:  # noqa: BLE001
        return None


def flush() -> None:
    lf = client()
    if lf is not None:
        with contextlib.suppress(Exception):
            lf.flush()


def push_prompts() -> list[str]:
    """Prompt management : chaque prompt versionné dans Langfuse, étiqueté avec PROMPT_VERSION."""
    from .agent import prompts
    lf = client()
    if lf is None:
        raise RuntimeError("clés Langfuse absentes de Monitoring_Agent/.env")
    out = []
    for name, text in (("agent4-investigator", prompts.INVESTIGATOR), ("agent4-diagnoser", prompts.DIAGNOSER),
                       ("agent4-judge", prompts.JUDGE)):
        p = lf.create_prompt(name=name, prompt=text, type="text", labels=["production"],
                             tags=["agent4", prompts.PROMPT_VERSION], commit_message=f"version {prompts.PROMPT_VERSION}")
        out.append(f"{name} v{p.version}")
    lf.flush()
    return out


def push_dataset(cases: list[dict], name: str = "agent4-diagnosis-test") -> int:
    """Les cas de test (bases jamais vues) deviennent un dataset Langfuse, rejouable en expérience."""
    lf = client()
    if lf is None:
        raise RuntimeError("clés Langfuse absentes de Monitoring_Agent/.env")
    with contextlib.suppress(Exception):
        lf.create_dataset(name=name, description="Agent 4 : diagnostics attendus sur car_loan / car_price")
    for c in cases:
        lf.create_dataset_item(dataset_name=name, id=c["id"], input={"messages": c["messages"]},
                               expected_output=c["oracle"], metadata={"base": c["base"], "injection": c["injection"]})
    lf.flush()
    return len(cases)
