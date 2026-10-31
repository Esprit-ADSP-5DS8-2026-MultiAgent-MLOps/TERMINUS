"""runs.jsonl -> jeux d'entraînement : SFT (trajectoires retenues) et préférences DPO (retenue vs écartée pour la MÊME tâche).

    python -m finetune.export --runs data/run1/runs.jsonl --out data/run1/export --per-task 1 --holdout-faults typo_import,syntax_error

Découpage PAR PROJET de base (jamais par exemple : sinon fuite entre train et test) ; les défauts « holdout » vont tous dans `test_unseen`.
Les valeurs qui ressemblent à des secrets (clés d'API de l'environnement) sont remplacées par [REDACTED].
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from finetune.recorder import redact, secret_values

SKIP_FOR_DPO = {"no_trajectory", "compacted_trajectory", "not_revalidated"}


def split_of(base_seed: int, val_pct: int = 10, test_pct: int = 10) -> str:
    h = int(hashlib.sha256(f"split:{base_seed}".encode()).hexdigest(), 16) % 100
    return "test" if h < test_pct else "val" if h < test_pct + val_pct else "train"


def _cost(run: dict) -> tuple:
    r = run["repair"]
    return (r["steps"], (run.get("trajectory") or {}).get("tokens", 0), r["added"] + r["removed"])


def _clean(obj, secrets: list[str]):
    return json.loads(redact(json.dumps(obj, ensure_ascii=False), secrets))


def load_runs(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def export(runs: list[dict], out_dir: Path, *, per_task: int = 1, holdout_faults=(), val_pct: int = 10, test_pct: int = 10, environ=None) -> dict:
    secrets = secret_values(environ)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in runs:
        groups[(r["base_seed"], r["fault_id"], r["run_id"].split("-")[2])].append(r)

    def where(run: dict) -> str:
        return "test_unseen" if run["fault_id"] in set(holdout_faults) else split_of(run["base_seed"], val_pct, test_pct)

    sft, dpo = collections.defaultdict(list), collections.defaultdict(list)
    for key, rs in groups.items():
        good = sorted((r for r in rs if r["accepted"]), key=_cost)[:per_task]
        for r in good:
            t = r["trajectory"]
            sft[where(r)].append({"id": r["run_id"], "messages": _clean(t["messages"], secrets), "tools": t["tools"],
                                  "meta": {"fault_id": r["fault_id"], "base_seed": r["base_seed"], "steps": r["repair"]["steps"], "tokens": t.get("tokens"),
                                           "model": r["repair"]["model"], "judge": (r.get("judge") or {}).get("verdict"), "files": r["repair"]["files_changed"]}})
        if good:
            ch = good[0]["trajectory"]["messages"]
            for bad in rs:
                bt = bad.get("trajectory")
                if bad["accepted"] or not bt or set(bad["rejected_by"]) & SKIP_FOR_DPO or bt["messages"][:2] != ch[:2]:
                    continue
                dpo[where(bad)].append({"id": f"{good[0]['run_id']}__vs__{bad['run_id']}", "prompt": _clean(ch[:2], secrets), "tools": good[0]["trajectory"]["tools"],
                                        "chosen": _clean(ch[2:], secrets), "rejected": _clean(bt["messages"][2:], secrets), "rejected_by": bad["rejected_by"],
                                        "fault_id": bad["fault_id"]})
    for name, items in (("sft", sft), ("dpo", dpo)):
        for split, rows in items.items():
            with (out_dir / f"{name}_{split}.jsonl").open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    reasons = collections.Counter(w.split(":")[0] for r in runs for w in r["rejected_by"])
    per_fault = collections.defaultdict(lambda: [0, 0])
    for r in runs:
        per_fault[r["fault_id"]][0] += 1
        per_fault[r["fault_id"]][1] += int(r["accepted"])
    acc = [r for r in runs if r["accepted"]]
    stats = {"runs": len(runs), "accepted": len(acc), "acceptance_rate": round(len(acc) / len(runs), 3) if runs else None,
             "tasks": len(groups), "tasks_with_a_good_run": sum(any(r["accepted"] for r in rs) for rs in groups.values()),
             "sft": {k: len(v) for k, v in sft.items()}, "dpo": {k: len(v) for k, v in dpo.items()},
             "rejection_reasons": dict(reasons.most_common()), "per_fault": {k: {"runs": v[0], "accepted": v[1]} for k, v in sorted(per_fault.items())},
             "mean_steps_accepted": round(sum(r["repair"]["steps"] for r in acc) / len(acc), 2) if acc else None,
             "holdout_faults": sorted(holdout_faults), "secrets_redacted_values": len(secrets)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True), ap.add_argument("--out", required=True), ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--holdout-faults", default="", help="défauts jamais vus à l'entraînement (liste séparée par des virgules)")
    ap.add_argument("--val-pct", type=int, default=10), ap.add_argument("--test-pct", type=int, default=10)
    a = ap.parse_args(argv)
    stats = export(load_runs(Path(a.runs)), Path(a.out), per_task=a.per_task, holdout_faults=[x for x in a.holdout_faults.split(",") if x],
                   val_pct=a.val_pct, test_pct=a.test_pct)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
