"""CLI : python -m repair_agent PROJET --report validator.json --output reparé/ [--json repair.json] [--attempt N] [--feedback judge.json]"""
import argparse
import sys
from pathlib import Path

from agentic_core.llm_client import build_client
from agentic_core.progress import ConsoleProgress, NullProgress

from .agent import RepairConfig, run_repair

EXIT = {"REPAIRED": 0, "NOTHING_TO_DO": 0, "PARTIAL": 10, "ESCALATED": 20, "FAILED": 30}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="repair_agent", description="Agent Repair : corrige dans une copie de travail isolée")
    ap.add_argument("project"), ap.add_argument("--report", required=True, help="rapport du Validator (JSON)")
    ap.add_argument("--output", required=True, help="dossier de la copie corrigée (nouveau, vide)")
    ap.add_argument("--json", help="écrit le rapport de réparation scellé"), ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--feedback", help="verdict REVISE du Judge (nouvelle tentative)")
    ap.add_argument("--model"), ap.add_argument("--env", default="venv"), ap.add_argument("--sandbox", default="auto", choices=["auto", "docker", "subprocess"])
    ap.add_argument("--no-exec", action="store_true", help="n'exécute ni import, ni tests, ni métriques (pas de sandbox)")
    ap.add_argument("--cache-dir"), ap.add_argument("--llm-cache"), ap.add_argument("--replay", action="store_true"), ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    client = build_client("repair", a.model, a.llm_cache, a.replay)
    rep = run_repair(a.project, a.report, a.output, client, RepairConfig(sandbox=a.sandbox, env=a.env, cache_dir=a.cache_dir, run_exec_checks=not a.no_exec),
                     attempt=a.attempt, judge_feedback=a.feedback, progress=NullProgress() if a.quiet else ConsoleProgress())
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n{rep.status}  (prochaine action : {rep.next_action} · approbation humaine : {rep.requires_human_approval})")
    for m in rep.modifications:
        print(f"  {m.patch_id} {m.action:<7} {m.file} [{m.risk}] — {m.reason}")
    for t in rep.tests:
        print(f"  test {t.name:<13} {t.status}")
    for f in rep.scope_flags:
        print(f"  ⚠ {f.kind} {f.file} {'(bloqué)' if f.blocked else ''} — {f.detail}")
    for u in rep.unresolved_issues:
        print(f"  non résolu {u.issue_id}{' (humain)' if u.needs_human else ''} — {u.reason}")
    if rep.failure:
        print(f"  ÉCHEC : {rep.failure.kind.value} — {rep.failure.message}")
    return EXIT[rep.status]


if __name__ == "__main__":
    sys.exit(main())
