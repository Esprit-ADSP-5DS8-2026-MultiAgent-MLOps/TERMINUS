"""Boucle d'agent générique, partagée par Validator, Repair et Judge.

Chaque agent fournit : son prompt, ses outils, son outil de dépôt (submit_*) et ses INVARIANTS (une fonction qui
valide la soumission et renvoie les problèmes à corriger). Aucune règle métier ici, seulement du protocole :
  - échec fermé : erreur LLM / quota / timeout / budget / trop d'étapes -> un STATUT d'échec, jamais une exception ;
  - une soumission refusée est renvoyée au LLM (avec les problèmes) un nombre limité de fois.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .llm_client import LLMError
from .progress import NullProgress
from .schemas import Failure, FailureKind

EMPTY_STATS = {"steps": 0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0,
               "llm_latency_s": 0.0, "submit_retries": 0, "replayed_calls": 0}


@dataclass
class LoopConfig:
    max_steps: int = 14
    token_budget: int = 60_000
    max_tool_calls_per_step: int = 4
    max_submit_retries: int = 2


@dataclass
class AgentSpec:
    name: str                                   # validator | repair | judge
    system_prompt: str
    user_message: str
    toolbox: Any                                # specs() -> list[dict] ; call(name, args) -> str
    submit_tool: str
    check_submission: Callable[[dict], tuple[Any, list[str]]]
    nudge: str = "Utilise les outils, puis dépose ton résultat."


@dataclass
class LoopResult:
    submission: Any = None
    problems: list[str] = field(default_factory=list)
    failure: Optional[Failure] = None
    stats: dict = field(default_factory=lambda: dict(EMPTY_STATS))


COMPACT_MIN_CHARS = 300
COMPLETION_RESERVE = 2300        # tokens à garder libres pour la réponse (rapport final + raisonnement)


def compact_messages(messages: list[dict], keep_last: int = 2) -> int:
    """Archive les anciennes sorties d'outils (et les très gros arguments d'appels refusés) pour alléger le contexte.
    Les `keep_last` sorties les plus récentes restent intactes. Retourne le nombre de messages réduits.
    Le LLM peut rappeler l'outil s'il a besoin d'un extrait exact pour une preuve (les preuves sont vérifiées contre TOUTES les sorties passées)."""
    names = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            names[c.get("id")] = (c.get("function") or {}).get("name", "outil")
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    done = 0
    for i in tool_idx[:max(0, len(tool_idx) - keep_last)]:
        c = messages[i].get("content") or ""
        if len(c) > COMPACT_MIN_CHARS and not c.startswith("[archivé"):
            messages[i] = {**messages[i], "content": f"[archivé] sortie de {names.get(messages[i].get('tool_call_id'), 'outil')} ({len(c)} car.) "
                                                      "retirée pour tenir dans la limite de tokens : rappelle l'outil si tu en as besoin (ex. pour citer une preuve exacte)."}
            done += 1
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and i != last_assistant:
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                if len(fn.get("arguments") or "") > 600 and "archivé" not in fn["arguments"]:
                    fn["arguments"] = json.dumps({"archivé": True, "note": "arguments volumineux retirés du contexte"})
                    done += 1
    return done


def _drop_nulls(x):
    """Un null pour un champ facultatif = champ omis (les schémas envoyés à Groq acceptent null, le reste du code non)."""
    if isinstance(x, dict):
        return {k: _drop_nulls(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_drop_nulls(v) for v in x if v is not None]
    return x


def _make_room(client, messages, tools) -> None:
    """Avant l'appel : si le contexte estimé laisse moins que COMPLETION_RESERVE tokens pour la réponse, on compacte progressivement."""
    limit, est = getattr(client, "tpm_limit", None), getattr(client, "prompt_estimate", None)
    if not limit or not est:
        return
    for keep in (3, 2, 1, 0):
        if est(messages, tools) <= limit - COMPLETION_RESERVE or not compact_messages(messages, keep):
            if est(messages, tools) <= limit - COMPLETION_RESERVE:
                return


def _fail(res: LoopResult, kind: FailureKind, msg: str) -> LoopResult:
    res.failure = Failure(kind=kind, message=msg)
    return res


def run_loop(client, spec: AgentSpec, cfg: LoopConfig, progress=None) -> LoopResult:
    progress = progress or NullProgress()
    res = LoopResult()
    st = res.stats
    messages = [{"role": "system", "content": spec.system_prompt}, {"role": "user", "content": spec.user_message}]
    nudges = 0
    progress.on_start(spec.name, f"(modèle {getattr(client, 'model', '?')})")
    try:
        for step in range(1, cfg.max_steps + 1):
            st["steps"] = step
            if st["total_tokens"] >= cfg.token_budget:
                return _fail(res, FailureKind.BUDGET_EXCEEDED, "budget de tokens dépassé")
            try:
                tools = spec.toolbox.specs()
                _make_room(client, messages, tools)
                try:
                    resp = client.chat(messages, tools)
                except LLMError as e:
                    if e.kind != "too_large":
                        raise
                    for keep in (1, 0):                                  # requête trop grande : on archive tout l'ancien et on réessaie
                        if compact_messages(messages, keep):
                            break
                    else:
                        raise
                    resp = client.chat(messages, tools)
            except LLMError as e:
                kind = {"quota": FailureKind.QUOTA, "too_large": FailureKind.QUOTA, "timeout": FailureKind.TIMEOUT}.get(e.kind, FailureKind.LLM_ERROR)
                hint = (" — réduisez le contexte ou augmentez la limite (GROQ_TPM_LIMIT, palier supérieur, modèle à plus grand quota)" if e.kind == "too_large" else "")
                return _fail(res, kind, f"erreur LLM ({e.kind}) : {e}{hint}")
            except Exception as e:  # un client défaillant ne doit jamais faire planter l'orchestrateur
                return _fail(res, FailureKind.LLM_ERROR, f"erreur LLM inattendue : {type(e).__name__}")
            st["llm_calls"] += 1
            st["replayed_calls"] += int(getattr(resp, "cached", False))
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
                st[k] += resp.usage.get(k, 0)
            st["llm_latency_s"] += resp.latency_s
            progress.on_llm(spec.name, step, resp.usage, resp.latency_s, getattr(resp, "model", ""))
            m = resp.message or {}
            calls = m.get("tool_calls") or []
            assistant = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if not calls:
                nudges += 1
                if nudges > 2:
                    return _fail(res, FailureKind.NO_TOOL_USE, "le LLM n'utilise pas les outils")
                messages.append({"role": "user", "content": spec.nudge})
                continue
            for i, call in enumerate(calls):
                fn = call.get("function") or {}
                name, cid = fn.get("name"), call.get("id") or f"call_{step}_{i}"
                try:
                    args, out = json.loads(fn.get("arguments") or "{}"), None
                    if not isinstance(args, dict):
                        raise json.JSONDecodeError("not an object", "", 0)
                    args = _drop_nulls(args)
                except json.JSONDecodeError:
                    args, out = None, json.dumps({"error": "arguments JSON invalides"})
                if args is not None and i >= cfg.max_tool_calls_per_step:
                    out = json.dumps({"error": f"maximum {cfg.max_tool_calls_per_step} appels d'outils par étape"})
                elif args is not None and name == spec.submit_tool:
                    sub, problems = spec.check_submission(args)
                    res.submission, res.problems = sub or res.submission, problems
                    if problems and st["submit_retries"] < cfg.max_submit_retries:
                        st["submit_retries"] += 1
                        progress.on_tool(spec.name, step, f"{spec.submit_tool} REFUSÉ :", "; ".join(problems)[:400])
                        out = json.dumps({"soumission_refusee": True, "problemes": problems,
                                          "instruction": f"corrige puis rappelle {spec.submit_tool} SANS relire de fichier : recopie à l'identique un extrait DÉJÀ présent dans les sorties d'outils de la conversation ; l'evidence d'un framework peut être omise."}, ensure_ascii=False)
                    else:
                        if sub is None:
                            _fail(res, FailureKind.INVALID_SUBMISSION, "aucune soumission valide : " + "; ".join(problems)[:300])
                        return res
                elif args is not None:
                    progress.on_tool(spec.name, step, name, json.dumps(args, ensure_ascii=False)[:80])
                    out = spec.toolbox.call(name, args)
                messages.append({"role": "tool", "tool_call_id": cid, "content": out})
        return _fail(res, FailureKind.MAX_STEPS, "nombre maximal d'étapes atteint")
    finally:
        progress.on_end(spec.name, st, res.failure)
