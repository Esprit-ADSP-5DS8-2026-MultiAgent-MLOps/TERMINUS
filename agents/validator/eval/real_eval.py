#!/usr/bin/env python
"""Faux rejets sur dépôts hétérogènes annotés À LA MAIN (eval/real/annotations.json). Aucune clé LLM requise par défaut.

    python eval/real_eval.py                       # validateur v2, partie déterministe (venv par projet : réseau requis)
    python eval/real_eval.py --env current         # interpréteur courant (rapide ; ENV.SANDBOX_ISOLATION restera NEEDS_REVIEW)
    python eval/real_eval.py --baseline            # ajoute la baseline v1 (validator/) pour comparer
    python eval/real_eval.py --llm                 # avec le LLM (GROQ_API_KEY)
    python eval/real_eval.py --only src_layout     # sous-ensemble

Mesures : FAUX REJET = dépôt jugé ACCEPTABLE par un humain mais classé REPAIR_REQUIRED/REJECTED.
          FAUX ACCEPTÉ = dépôt NOT_ACCEPTABLE classé ACCEPTED.  Au niveau des contrôles : FAIL inattendus et défauts manqués.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

REJECTS = {"REPAIR_REQUIRED", "REJECTED"}


def run_v2(path, a):
    from agentic_core.llm_client import ScriptedClient, build_client
    from agentic_core.schemas import CheckStatus
    from validator_agent import ValidatorConfig, run_validator
    if a.llm:
        client, cfg = build_client("validator", a.model, a.llm_cache, False), ValidatorConfig(env=a.env)
    else:
        client, cfg = ScriptedClient([]), ValidatorConfig(env=a.env, max_llm_steps=0)
    rep = run_validator(path, client, "sklearn-tabular", cfg)
    fails = sorted({c.check_id for c in rep.checks if c.status == CheckStatus.FAIL})
    open_ = sorted({c.check_id for c in rep.checks if c.status == CheckStatus.NOT_CHECKED})
    return rep.status.value, fails, open_


def run_v1(path, a):
    from validator import validate
    return validate(path, bandit=False, env=a.env).verdict.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="venv")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model"), ap.add_argument("--llm-cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    ann = {k: v for k, v in json.loads((HERE / "real" / "annotations.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    rows = []
    for name, c in sorted(ann.items()):
        if a.only and not any(o in name for o in a.only):
            continue
        path = HERE / "real" / "repos" / name
        if not path.is_dir():
            print(f"(absent : {path})", file=sys.stderr)
            continue
        status, fails, open_ = run_v2(path, a)
        row = {"repo": name, "human": c["human_verdict"], "acceptable": c["acceptable_statuses"], "status": status,
               "fails": fails, "not_checked": open_,
               "known_gap_fails": sorted(set(fails) & set(c.get("known_gaps", {}))),
               "unexpected_fails": sorted(set(fails) - set(c["expected_fail_checks"]) - set(c.get("known_gaps", {}))),
               "missed_defects": sorted(set(c["expected_fail_checks"]) - set(fails)),
               "status_ok": status in c["acceptable_statuses"],
               "false_reject": c["human_verdict"] == "ACCEPTABLE" and status in REJECTS,
               "false_accept": c["human_verdict"] != "ACCEPTABLE" and status == "ACCEPTED"}
        if a.baseline:
            row["v1"] = run_v1(path, a)
        rows.append(row)
        print(f"  {name:26} {c['human_verdict']:14} -> {status:16} {'OK' if row['status_ok'] else 'ÉCART'}", flush=True)

    ok_rows = [r for r in rows if r["human"] == "ACCEPTABLE"]
    print("\n| Dépôt | Jugement humain | Statut v2 |" + (" Baseline v1 |" if a.baseline else "") + " FAIL inattendus | Défauts manqués |")
    print("|---|---|---|" + ("---|" if a.baseline else "") + "---|---|")
    for r in rows:
        flag = " ✖ FAUX REJET" if r["false_reject"] else (" ✖ FAUX ACCEPTÉ" if r["false_accept"] else ("" if r["status_ok"] else " ≠"))
        print(f"| {r['repo']} | {r['human']} | {r['status']}{flag} |" + (f" {r['v1']} |" if a.baseline else "")
              + f" {', '.join(r['unexpected_fails']) or '-'} | {', '.join(r['missed_defects']) or '-'} |")
    fr = sum(r["false_reject"] for r in rows)
    print(f"\nFAUX REJETS : {fr}/{len(ok_rows)} dépôts acceptables"
          + (f"   |   baseline v1 : {sum(r['v1'] == 'REJECTED' for r in ok_rows)}/{len(ok_rows)}" if a.baseline else ""))
    print(f"FAUX ACCEPTÉS : {sum(r['false_accept'] for r in rows)}/{len([r for r in rows if r['human'] != 'ACCEPTABLE'])} dépôts non acceptables")
    print(f"Statut conforme au jugement humain : {sum(r['status_ok'] for r in rows)}/{len(rows)}")
    print(f"Défauts attendus manqués : {sum(len(r['missed_defects']) for r in rows)}   |   FAIL inattendus : {sum(len(r['unexpected_fails']) for r in rows)}"
          f"   |   lacunes connues encore présentes : {sum(len(r['known_gap_fails']) for r in rows)}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
