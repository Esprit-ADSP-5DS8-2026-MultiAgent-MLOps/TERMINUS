"""CLI : python -m repair_agent.judge PROJET --repair repair.json [--validator validator.json] [--json verdict.json]"""
import argparse
import sys
from pathlib import Path

from agentic_core.llm_client import build_client
from agentic_core.progress import ConsoleProgress, NullProgress

from .evaluator import JudgeConfig, run_judge

EXIT = {"ACCEPT": 0, "REVISE": 10, "REJECT": 20}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="repair_agent.judge", description="Agent Judge : verdict sur une réparation (jamais une décision d'admission)")
    ap.add_argument("project", help="projet SOURCE (original)"), ap.add_argument("--repair", required=True, help="rapport de réparation (JSON)")
    ap.add_argument("--validator", help="rapport du Validator d'origine (JSON)"), ap.add_argument("--workspace", help="copie corrigée (défaut : celle du rapport)")
    ap.add_argument("--json", help="écrit le verdict scellé"), ap.add_argument("--model")
    ap.add_argument("--require-distinct-model", action="store_true")
    ap.add_argument("--llm-cache"), ap.add_argument("--replay", action="store_true"), ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    client = build_client("judge", a.model, a.llm_cache, a.replay)
    v = run_judge(a.project, a.repair, client, a.validator, JudgeConfig(require_distinct_model=a.require_distinct_model), workspace=a.workspace,
                  progress=NullProgress() if a.quiet else ConsoleProgress())
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(v.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n{v.verdict}  → {v.next_action}  (modèle indépendant du Repair : {v.independent_model})")
    for c in v.criteria:
        print(f"  {c.name:<17} {c.result:<7} {c.rationale[:110]}")
    for c in v.remaining_concerns:
        print(f"  préoccupation : {c}")
    if v.overridden_by_invariant:
        print(f"  ⚠ ACCEPT refusé par l'infrastructure : {v.overridden_by_invariant}")
    if v.failure:
        print(f"  ÉCHEC FERMÉ : {v.failure.kind.value} — {v.failure.message}")
    return 30 if v.failure else EXIT[v.verdict]


if __name__ == "__main__":
    sys.exit(main())
