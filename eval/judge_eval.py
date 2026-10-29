#!/usr/bin/env python
"""Mesure le Judge : il peut se tromper. Fausse acceptation d'une mauvaise réparation = la mesure la plus grave.

    GROQ_API_KEY=... python eval/judge_eval.py --out results_judge.json
    python eval/judge_eval.py --llm-cache ~/.cache/agentic-mlops/judge --replay      # rejeu, sans appeler Groq
    python eval/judge_eval.py --scope-only                                           # garde de périmètre seule (sans LLM)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from metrics import score_judge  # noqa: E402

from agentic_core.schemas import RepairReport, ScopeFlag, seal  # noqa: E402
from repair_agent.runner import run_checks  # noqa: E402
from repair_agent.scope import ScopeGuard  # noqa: E402
from validator.handoff import compute_project_hash  # noqa: E402

CASES = HERE / "judge_cases"


def load_cases(only=None):
    if not CASES.exists():
        import make_judge_cases
        make_judge_cases.build()
    out = {}
    for d in sorted(CASES.iterdir()):
        if d.is_dir() and (not only or any(o in d.name for o in only)):
            out[d.name] = json.loads((d / "case.json").read_text())
    return out


def synthetic_repair_report(case_dir: Path) -> RepairReport:
    """Rapport de réparation « tel qu'un Repair honnête le produirait » : modifications = diff réel ; tests relancés pour de vrai (sans exécution)."""
    from repair_agent.judge.tools import _read
    from agentic_core.schemas import Modification
    from repair_agent.patches import unified
    from validator.fs import walk_project
    src, rep = case_dir / "original", case_dir / "repaired"
    a = {f.as_posix() for f in walk_project(src)[0]}
    b = {f.as_posix() for f in walk_project(rep)[0]}
    mods, diff = [], ""
    for rel in sorted(a | b):
        before = _read(src / rel) if rel in a else None
        after = _read(rep / rel) if rel in b else None
        if before == after:
            continue
        mods.append(Modification(patch_id=f"p{len(mods) + 1:04d}", file=rel, action="create" if before is None else "delete" if after is None else "rewrite", issue_id="I1", reason="réparation annotée"))
        diff += unified(rel, before, after)
    tests = run_checks(rep, None)
    return seal(RepairReport(repair_id=f"eval-{case_dir.name}", project=case_dir.name, status="REPAIRED", source_project_hash=compute_project_hash(src)[0],
                             repaired_project_hash=compute_project_hash(rep)[0], workspace_path=str(rep), modifications=mods, tests=tests, diff=diff,
                             requires_judge=True, requires_revalidation=True, next_action="JUDGE", llm={"model": "eval-repair"}))


def scope_scan(case_dir: Path) -> set[str]:
    """Ce que la garde de périmètre déterministe détecte seule (sans LLM) : même logique que celle du Judge."""
    from repair_agent.judge.tools import JudgeToolBox
    rr = synthetic_repair_report(case_dir)
    tb = JudgeToolBox(case_dir / "original", case_dir / "repaired", rr)
    return {f.kind for f in tb.flags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*"), ap.add_argument("--model"), ap.add_argument("--out")
    ap.add_argument("--llm-cache"), ap.add_argument("--replay", action="store_true"), ap.add_argument("--scope-only", action="store_true")
    a = ap.parse_args()
    cases, results = load_cases(a.only), []
    for name, c in cases.items():
        d = CASES / name
        flags = scope_scan(d)
        row = {"name": name, "label": c["label"], "scope_flags": sorted(flags), "expected_scope_flags": c["expected_scope_flags"], "needs_llm": c["needs_llm"]}
        if a.scope_only:
            row["verdict"] = "REVISE" if flags else "ACCEPT"            # sans LLM : le périmètre seul, borne basse
        else:
            from agentic_core.llm_client import build_client
            from repair_agent.judge import run_judge
            rr = synthetic_repair_report(d)
            v = run_judge(d / "original", rr, build_client("judge", a.model, a.llm_cache, a.replay), workspace=d / "repaired")
            row.update(verdict=v.verdict, next_action=v.next_action, failed_closed=bool(v.failure), overridden=v.overridden_by_invariant,
                       tokens=v.llm.get("total_tokens"), criteria={c.name: c.result for c in v.criteria})
        results.append(row)
        print(f"{name:26} {c['label']:5} -> {row['verdict']:7} flags={row['scope_flags']}", file=sys.stderr)
    summary = score_judge(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
