"""
Évaluation du diagnostiqueur sur les cas de TEST (bases jamais vues : car_loan, car_price).

Même prompt pour tous (faits seuls), même contrôle que l'agent :
- JSON valide (conforme au schéma) ;
- bonne cause, bonne action (oracle = procédure) ;
- preuves vérifiées (guards.verify) ;
- action DANGEREUSE : réentraîner quand il ne faut pas (le pire défaut d'un agent de monitoring) ;
- latence et jetons.

Comparaison à 4 voies : règles (100 % par construction : c'est l'oracle — sert de plafond « procédure »),
Qwen de base, Qwen fine-tuné, grand LLM (Groq). Le fine-tuning vaut le coup s'il rapproche le petit modèle
local du grand, pour une fraction du coût et sans envoyer les données hors de la machine.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..agent.guards import verify
from ..agent.llm import LLMError, _settings, build_model, structured, usage_summary
from ..agent.schemas import Diagnosis


def evaluate_model(spec: str, cases_path: Path, limit: int | None = None, progress: Callable[[str], None] = print) -> dict:
    model = build_model(spec, _settings())
    if model is None:
        raise LLMError(f"clé manquante pour {spec}")
    cases = [json.loads(x) for x in cases_path.read_text(encoding="utf-8").splitlines() if x.strip()][:limit]
    rows = []
    for i, c in enumerate(cases, 1):
        meter: list = []
        t0 = time.perf_counter()
        try:
            d = structured(model, c["messages"], Diagnosis, temperature=0.0, meter=meter)
            v = verify(d, c["facts"], c["features"], signals=set(c["signals"]))
            row = {"id": c["id"], "valid_json": True, "cause_ok": d.root_cause == c["oracle"]["root_cause"],
                   "action_ok": d.action == c["oracle"]["action"], "evidence_ok": v.ok,
                   "unsafe": d.action in ("request_retraining", "rollback_model") and d.action != c["oracle"]["action"],
                   "action": d.action, "expected": c["oracle"]["action"]}
        except LLMError as exc:
            row = {"id": c["id"], "valid_json": False, "cause_ok": False, "action_ok": False, "evidence_ok": False,
                   "unsafe": False, "action": None, "expected": c["oracle"]["action"], "error": str(exc)[:200]}
        row["seconds"] = round(time.perf_counter() - t0, 2)
        row["tokens"] = usage_summary(meter)["completion_tokens"]
        rows.append(row)
        progress(f"  [{i}/{len(cases)}] {c['id']} : {row['action']} (attendu {row['expected']}) {row['seconds']} s")
    n = len(rows) or 1

    def rate(k: str) -> float:
        return round(sum(r[k] for r in rows) / n, 3)

    return {"model": spec, "cases": len(rows), "valid_json": rate("valid_json"), "cause_accuracy": rate("cause_ok"),
            "action_accuracy": rate("action_ok"), "evidence_verified": rate("evidence_ok"), "unsafe": rate("unsafe"),
            "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
            "confusion": dict(Counter(f"{r['expected']} -> {r['action']}" for r in rows if not r["action_ok"])),
            "rows": rows}


def report(results: list[dict]) -> str:
    lines = ["| modèle | cas | JSON valide | cause | action | preuves vérifiées | dangereuses | latence |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['cases']} | {r['valid_json']:.0%} | {r['cause_accuracy']:.0%} | "
                     f"{r['action_accuracy']:.0%} | {r['evidence_verified']:.0%} | {r['unsafe']:.0%} | {r['mean_seconds']} s |")
    return "\n".join(lines) + "\n"
