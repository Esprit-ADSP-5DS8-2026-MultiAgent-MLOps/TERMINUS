#!/usr/bin/env python
"""Mesure la chaîne Validator -> Repair -> Judge -> revalidation sur les projets annotés REPAIR_REQUIRED (taux de réparation, tentatives, régressions).

    GROQ_API_KEY=... python eval/repair_eval.py --max-attempts 2 --out results_repair.json
Séquence simple : ce n'est PAS l'orchestrateur (hors périmètre), seulement de quoi mesurer les agents.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from compare import load_cases  # noqa: E402
from metrics import score_repair  # noqa: E402


def chain(path, case, a, tmp):
    from agentic_core.llm_client import build_client
    from repair_agent.judge import run_judge
    from repair_agent import RepairConfig, run_repair
    from validator_agent import ValidatorConfig, run_validator
    vcfg = ValidatorConfig(env=a.env)
    v = run_validator(path, build_client("validator", a.model, a.llm_cache, a.replay), case["profile"], vcfg)
    row = {"name": case["name"], "initial_status": v.status.value, "repair_status": "SKIPPED", "attempts": 0, "revalidated_status": None, "regressions": 0}
    feedback = None
    for attempt in range(1, a.max_attempts + 1):
        r = run_repair(path, v, Path(tmp) / f"{case['name']}_a{attempt}", build_client("repair", a.model, a.llm_cache, a.replay),
                       RepairConfig(env=a.env), attempt=attempt, judge_feedback=feedback)
        row.update(repair_status=r.status, attempts=attempt)
        if r.next_action != "JUDGE":
            break
        j = run_judge(path, r, build_client("judge", None, a.llm_cache, a.replay), v)
        row["judge_verdict"] = j.verdict
        if j.verdict == "ACCEPT":
            v2 = run_validator(r.workspace_path, build_client("validator", a.model, a.llm_cache, a.replay), case["profile"], vcfg, parent=j)
            row["revalidated_status"] = v2.status.value
            before = {i.check_id for i in v.issues}
            row["regressions"] = len({i.check_id for i in v2.issues} - before) + int(bool(r.metrics_comparison and r.metrics_comparison.behavior_changed))
            break
        if j.next_action != "RETRY":
            break
        feedback = j
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*"), ap.add_argument("--max-attempts", type=int, default=2), ap.add_argument("--model")
    ap.add_argument("--env", default="venv"), ap.add_argument("--llm-cache"), ap.add_argument("--replay", action="store_true"), ap.add_argument("--out")
    a = ap.parse_args()
    if not a.replay and not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY manquante (ou utilisez --replay).", file=sys.stderr)
        return 2
    cases = {k: dict(v, name=k) for k, v in load_cases(a.only).items() if v["expected_status"] == "REPAIR_REQUIRED"}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, c in sorted(cases.items()):
            rows.append(chain(HERE / c["path"], c, a, tmp))
            print(rows[-1], file=sys.stderr)
    summary = score_repair(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
