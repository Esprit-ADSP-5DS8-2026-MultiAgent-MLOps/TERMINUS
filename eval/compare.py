#!/usr/bin/env python
"""Compare le validateur DÉTERMINISTE (baseline, validator/) et le VALIDATOR AGENT v2 sur le jeu annoté (eval/annotations.json).

    python eval/compare.py --mode baseline                                   # baseline (construit des venvs : réseau requis)
    GROQ_API_KEY=... python eval/compare.py --mode agent --runs 3 --out results_agent.json
    python eval/compare.py --mode agent --llm-cache ~/.cache/agentic-mlops/v --replay    # rejeu sans appeler Groq
    python eval/compare.py --mode agent --split holdout                      # jeu JAMAIS vu uniquement

Les deux sont ramenés à la même échelle : codes de la baseline -> contrôles du catalogue -> statut calculé par le MÊME moteur de politique.
Métriques : rappel, précision, taux de fausses alertes, fiabilité des décisions, et le FAUX ACCEPTÉ (la mesure la plus grave).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from code_map import check_for  # noqa: E402
from metrics import score_validator  # noqa: E402

from agentic_core.schemas import Check, CheckStatus  # noqa: E402
from policy import load_contract  # noqa: E402
from policy.engine import evaluate  # noqa: E402


def load_cases(only=None, split=None):
    for script, marker in (("make_variants.py", "projects/variants"), ("make_adversarial.py", "projects/adversarial"), ("make_multi.py", "projects/multi")):
        if not (HERE / marker).exists():
            subprocess.run([sys.executable, str(HERE / script)], check=True)
    if not (HERE / "annotations.json").exists():
        subprocess.run([sys.executable, str(HERE / "make_annotations.py")], check=True)
    cases = json.loads((HERE / "annotations.json").read_text())
    return {k: v for k, v in cases.items() if (not only or any(o in k for o in only)) and (not split or v["split"] == split)}


def run_baseline(path, case, args):
    """Baseline déterministe -> contrôles v2 -> statut par le moteur de politique. Hors profil sklearn-tabular : non comparable."""
    if case["profile"] != "sklearn-tabular":
        return None
    from validator import validate
    rep = validate(path, bandit=False, env=args.env)
    failed = {cid for f in rep.findings if f.severity.value != "INFO" and (cid := check_for(f.code))}
    contract = load_contract(case["profile"])
    cs = [Check(check_id=i, status=CheckStatus.FAIL if i in failed else CheckStatus.PASS, evidence="x") for i in contract.mandatory_ids]
    cs += [Check(check_id=i, status=CheckStatus.FAIL, evidence="x") for i in failed if contract.check(i) and i not in contract.mandatory_ids]
    pol, _ = evaluate(contract, cs, [], executed_ok=True)
    return pol.status.value, sorted(failed), {"tokens": 0, "seconds": rep.duration_s}


def run_agent(path, case, args):
    from agentic_core.llm_client import build_client
    from validator_agent import ValidatorConfig, run_validator
    rep = run_validator(path, build_client("validator", args.model, args.llm_cache, args.replay), case["profile"], ValidatorConfig(env=args.env))
    failed = sorted({c.check_id for c in rep.checks if c.status == CheckStatus.FAIL})
    return rep.status.value, failed, {"tokens": rep.llm.get("total_tokens", 0), "cached": rep.llm.get("cached_tokens", 0), "seconds": rep.duration_s,
                                      "failed_closed": rep.run_status.value == "FAILED_CLOSED", "limits": len(rep.limits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "agent"], required=True)
    ap.add_argument("--runs", type=int, default=1, help="lancements par projet (mesure la stabilité)")
    ap.add_argument("--only", nargs="*"), ap.add_argument("--split", choices=["dev", "holdout"])
    ap.add_argument("--model"), ap.add_argument("--env", default="venv"), ap.add_argument("--llm-cache"), ap.add_argument("--replay", action="store_true")
    ap.add_argument("--out", help="résultats détaillés (JSON)")
    a = ap.parse_args()
    if a.mode == "agent" and not a.replay and not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY manquante (ou utilisez --replay).", file=sys.stderr)
        return 2
    run = run_baseline if a.mode == "baseline" else run_agent
    cases, rows, results = load_cases(a.only, a.split), [], []
    for name, c in sorted(cases.items()):
        outs = [run(HERE / c["path"], c, a) for _ in range(a.runs)]
        if outs[0] is None:
            continue
        statuses = [o[0] for o in outs]
        modal, n_modal = Counter(statuses).most_common(1)[0]
        results.append({"name": name, "split": c["split"], "expected_status": c["expected_status"], "predicted_status": modal,
                        "expected_checks": c["expected_checks"], "predicted_checks": sorted({x for o in outs for x in o[1]}) if a.runs == 1 else outs[statuses.index(modal)][1]})
        rows.append({"project": name, "expected": c["expected_status"], "statuses": statuses, "stability": n_modal / len(statuses),
                     "tokens": sum(o[2]["tokens"] for o in outs), "seconds": round(sum(o[2]["seconds"] for o in outs), 1), "meta": [o[2] for o in outs]})
    summary = score_validator(results)
    print(f"\nMode : {a.mode}   projets : {len(results)}   lancements/projet : {a.runs}\n")
    print("| Projet | Split | Attendu | Obtenu | Stabilité |")
    print("|---|---|---|---|---|")
    for r, x in zip(rows, results):
        got = x["predicted_status"] if len(set(r["statuses"])) == 1 else " / ".join(f"{k}×{v}" for k, v in Counter(r["statuses"]).items())
        mark = "✖ FAUX ACCEPTÉ" if x["expected_status"] != "ACCEPTED" and x["predicted_status"] == "ACCEPTED" else ("✔" if got == r["expected"] else "≠")
        print(f"| {r['project']} | {x['split']} | {r['expected']} | {got} {mark} | {r['stability']:.0%} |")
    for k, s in summary.items():
        print(f"\n[{k}] n={s['n']}  rappel={s['issue_recall']}  précision={s['issue_precision']}  fausses alertes={s['false_alarm_rate']}  fiabilité décisions={s['decision_reliability']}"
              f"\n       FAUX ACCEPTÉS={s['false_accepts']} (taux {s['false_accept_rate']})  faux rejets={s['false_rejects']}  taux NEEDS_REVIEW={s['needs_review_rate']}")
    if rows:
        print(f"\nStabilité moyenne : {statistics.mean(r['stability'] for r in rows):.0%}   |   tokens : {sum(r['tokens'] for r in rows)}   |   durée : {sum(r['seconds'] for r in rows):.0f}s")
    if a.out:
        Path(a.out).write_text(json.dumps({"mode": a.mode, "summary": summary, "rows": rows, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
