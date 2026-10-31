"""
Distillation vérifiée : de grands modèles (Groq) rédigent les diagnostics que le petit modèle apprendra.

Pour chaque scénario (prompt identique à celui que verra le SLM : faits seuls, sans enquête ni RAG) :
1. le professeur répond à l'aveugle ;
2. on garde sa réponse seulement si : bonne cause, bonne action (oracle) et toutes les preuves
   vérifiées par le vérificateur de l'agent (guards.verify, le même qu'en production) ;
3. sinon rationalisation (STaR, Zelikman et al. 2022) : on lui donne la bonne réponse en indice et on lui
   demande de la justifier ; même contrôle. L'exemple d'entraînement garde le prompt sans l'indice :
   le petit modèle apprend à trouver la réponse seul ;
4. sinon : pas d'exemple « professeur » (build.py peut utiliser le diagnostic des règles en repli).

Plusieurs professeurs en rotation (quotas séparés, styles variés). Reprenable : chaque scénario traité est
écrit immédiatement dans teacher.jsonl ; relancer la commande reprend où on s'était arrêté.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from ..agent.guards import verify
from ..agent.llm import ChatModel, LLMError, LLMUnavailable, _settings, build_model, structured, usage_summary
from ..agent.schemas import Diagnosis

DEFAULT_TEACHERS = "groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b,groq:openai/gpt-oss-20b"

HINT = ("<indice>\nLa procédure, appliquée aux faits, donne : root_cause = {cause}, action = {action}.\n"
        "Rédige le diagnostic qui le JUSTIFIE, en citant uniquement des clés de <faits> avec leur valeur exacte.\n"
        "</indice>")


def teachers(spec: str | None = None) -> list[ChatModel]:
    s = _settings()
    specs = (spec or s.get("MONITORING_TEACHERS") or DEFAULT_TEACHERS).split(",")
    models = [build_model(x.strip(), s) for x in specs if x.strip()]
    models = [m for m in models if m is not None]
    if not models:
        raise LLMError("aucun professeur utilisable (GROQ_API_KEY dans le .env ?)")
    return models


def check(d: Diagnosis, sc: dict) -> list[str]:
    problems = []
    if d.root_cause != sc["oracle"]["root_cause"]:
        problems.append(f"cause {d.root_cause} ≠ {sc['oracle']['root_cause']}")
    if d.action != sc["oracle"]["action"]:
        problems.append(f"action {d.action} ≠ {sc['oracle']['action']}")
    v = verify(d, sc["facts"], sc["features"], signals=set(sc["signals"]))
    return problems + v.problems


def normalize(d: Diagnosis, facts: dict) -> dict:
    """Valeurs recopiées exactement (le vérificateur tolère 2 % ; l'exemple d'entraînement, non)."""
    out = json.loads(d.model_dump_json())
    for ev in out["evidence"]:
        ev["value"] = facts[ev["fact"]]
    out["confidence"] = round(float(out["confidence"]), 2)
    return out


def order(scenarios: list[dict]) -> list[dict]:
    """Tourniquet sur (action attendue, base) : si le quota s'épuise, chaque classe a déjà des exemples."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in scenarios:
        groups[(s["oracle"]["action"], s["base"])].append(s)
    out, queues = [], list(groups.values())
    while queues:
        queues = [q for q in queues if q]
        out += [q.pop(0) for q in queues]
    return out


WAIT = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)(ms|s))?", re.IGNORECASE)


def quota_wait(message: str) -> tuple[float, bool]:
    """(secondes à attendre, quota journalier ?) d'après le message 429 de Groq.

    Les quotas gratuits sont surtout par minute (jetons/min, requêtes/min) : il suffit d'attendre un peu.
    Un quota par jour avec une longue attente met le professeur de côté jusqu'à la prochaine session."""
    m = WAIT.search(message or "")
    wait = 60.0
    if m and any(m.groups()):
        h, mi, sec, unit = m.groups()
        wait = 3600 * int(h or 0) + 60 * int(mi or 0) + (float(sec or 0) / (1000 if unit == "ms" else 1))
    daily = "per day" in (message or "").lower()
    return max(wait, 2.0), daily and wait > 900


def teach(scenarios_path: Path, out: Path, spec: str | None = None, max_calls: int = 400,
          splits: tuple[str, ...] = ("train", "valid"), progress: Callable[[str], None] = print,
          sleep: Callable[[float], None] = time.sleep) -> Counter:
    models = teachers(spec)
    for m in models:
        # On gère nous-mêmes les quotas (rotation + attente) : le client ne doit pas bloquer longtemps.
        if hasattr(m, "retries"):
            m.retries, m.max_wait_s = 0, 0
        if "gpt-oss" in getattr(m, "model", ""):
            m.reasoning_effort = "low"        # moins de jetons de réflexion : la réponse est vérifiée de toute façon
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in (json.loads(x) for x in scenarios_path.read_text(encoding="utf-8").splitlines() if x.strip())
            if s["split"] in splits and s["id"] not in done]
    queue = order(todo)
    stats: Counter = Counter()
    exhausted: set[str] = set()
    ready_at: dict[str, float] = {m.name: 0.0 for m in models}
    calls, turn = 0, 0
    progress(f"  {len(queue)} scénarios à traiter, professeurs : {', '.join(m.name for m in models)}")
    with out.open("a", encoding="utf-8") as f:
        while queue and calls < max_calls:
            live = [m for m in models if m.name not in exhausted]
            if not live:
                break
            now = time.monotonic()
            free = [m for m in live if ready_at[m.name] <= now]
            if not free:                       # tous en pause (quota par minute) : on attend le premier libre
                sleep(min(ready_at[m.name] for m in live) - now + 0.5)
                continue
            model = free[turn % len(free)]
            turn += 1
            sc = queue[0]
            meter: list = []
            record = {"id": sc["id"], "model": model.name, "source": None, "diagnosis": None, "problems": []}
            try:
                for attempt, messages in (("teacher", sc["messages"]),
                                          ("teacher_rationalized", None)):
                    if messages is None:     # 2e essai : l'indice est ajouté au dernier message utilisateur
                        hint = HINT.format(cause=sc["oracle"]["root_cause"], action=sc["oracle"]["action"])
                        messages = sc["messages"][:-1] + [{"role": "user",
                                                           "content": sc["messages"][-1]["content"] + "\n\n" + hint}]
                    calls += 1
                    try:
                        d = structured(model, messages, Diagnosis, temperature=0.2, meter=meter)
                    except LLMUnavailable:
                        raise
                    except LLMError as exc:
                        record["problems"].append(f"{attempt}: {str(exc)[:160]}")
                        continue
                    problems = check(d, sc)
                    if not problems:
                        record.update(source=attempt, diagnosis=normalize(d, sc["facts"]))
                        break
                    record["problems"].append(f"{attempt}: " + "; ".join(problems)[:300])
            except LLMUnavailable as exc:
                wait, daily = quota_wait(str(exc))
                if daily:
                    exhausted.add(model.name)
                    progress(f"  {model.name} : quota JOURNALIER atteint (reprise dans {wait / 60:.0f} min) — on continue sans lui")
                else:
                    ready_at[model.name] = time.monotonic() + wait
                    stats["pauses"] += 1
                continue                       # le même scénario sera repris par le prochain professeur libre
            queue.pop(0)
            record["usage"] = usage_summary(meter)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            stats[record["source"] or "rejected"] += 1
            n = sum(v for k, v in stats.items() if k != "pauses")
            if n % 10 == 0:
                progress(f"  {n} traités · {dict(stats)} · {calls} appels · reste {len(queue)}")
    progress(f"  fin : {dict(stats)} · {calls} appels · reste {len(queue)} · "
             f"quota journalier atteint : {sorted(exhausted) or 'aucun'}")
    return stats
