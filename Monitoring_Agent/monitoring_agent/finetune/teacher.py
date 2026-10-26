"""
Distillation VÉRIFIÉE : de grands modèles (Groq) rédigent les diagnostics que le petit modèle apprendra.

Pour chaque scénario (prompt identique à celui que verra le SLM : faits seuls, sans enquête ni RAG) :
1. le professeur répond À L'AVEUGLE ;
2. on garde sa réponse seulement si : bonne cause ET bonne action (oracle) ET toutes les preuves
   vérifiées par le vérificateur de l'agent (guards.verify, le même qu'en production) ;
3. sinon RATIONALISATION (STaR, Zelikman et al. 2022) : on lui donne la bonne réponse en indice et on lui
   demande de la justifier ; même contrôle. L'exemple d'entraînement garde le prompt SANS l'indice :
   le petit modèle apprend à trouver la réponse seul ;
4. sinon : pas d'exemple « professeur » (build.py peut utiliser le diagnostic des règles en repli).

Plusieurs professeurs en rotation (quotas séparés, styles variés). Reprenable : chaque scénario traité est
écrit immédiatement dans teacher.jsonl ; relancer la commande reprend où on s'était arrêté.
"""

from __future__ import annotations

import json
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
        raise LLMError("aucun professeur utilisable (GROQ_API_KEY dans Monitoring_Agent/.env ?)")
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
    """Valeurs recopiées EXACTEMENT (le vérificateur tolère 2 % ; l'exemple d'entraînement, non)."""
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


def teach(scenarios_path: Path, out: Path, spec: str | None = None, max_calls: int = 400,
          splits: tuple[str, ...] = ("train", "valid"), progress: Callable[[str], None] = print) -> Counter:
    models = teachers(spec)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in (json.loads(x) for x in scenarios_path.read_text(encoding="utf-8").splitlines() if x.strip())
            if s["split"] in splits and s["id"] not in done]
    todo = order(todo)
    stats: Counter = Counter()
    exhausted: set[str] = set()
    calls, turn = 0, 0
    progress(f"  {len(todo)} scénarios à traiter, professeurs : {', '.join(m.name for m in models)}")
    with out.open("a", encoding="utf-8") as f:
        for sc in todo:
            live = [m for m in models if m.name not in exhausted]
            if not live or calls >= max_calls:
                break
            model = live[turn % len(live)]
            turn += 1
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
                exhausted.add(model.name)
                progress(f"  {model.name} indisponible (quota ?) : {str(exc)[:140]} — on continue sans lui")
                continue
            record["usage"] = usage_summary(meter)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            stats[record["source"] or "rejected"] += 1
            n = sum(stats.values())
            if n % 10 == 0:
                progress(f"  {n} traités · {dict(stats)} · {calls} appels")
            time.sleep(0.2)
    progress(f"  fin : {dict(stats)} · {calls} appels · épuisés : {sorted(exhausted) or 'aucun'}")
    return stats
