"""
Ligne de commande de l'orchestrateur.

    python -m orchestrator run eval/projects/credit_risk            # interactif
    python -m orchestrator run eval/projects/credit_risk --no-wait  # s'arrête à la 1re pause
    python -m orchestrator status                                   # liste des runs
    python -m orchestrator status <run_id>                          # détail d'un run
    python -m orchestrator approve <run_id> [-m "commentaire"]
    python -m orchestrator reject  <run_id> [-m "commentaire"]
    python -m orchestrator resume  <run_id>                         # après un plantage

L'état de chaque run est sauvegardé dans runs/checkpoints.sqlite :
un run en pause peut être approuvé plus tard, depuis un autre terminal.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from .checkpoint import describe_backend
from .graph import Run, RunNotFound, list_runs

ICONS = {"OK": "✔", "WARNING": "!", "NEEDS_APPROVAL": "?", "FAILED": "✖"}
STATUS_TEXT = {
    "WAITING_HUMAN": "en pause, attend une décision humaine",
    "INTERRUPTED": "arrêté au milieu, à reprendre avec `resume`",
}


def ask_human(question: dict) -> dict:
    print(f"\n── Validation humaine : {question['gate']} ──")
    print(question["question"])
    if question.get("file_to_review"):
        print(f"Fichier à relire : {question['file_to_review']}")
    while True:
        answer = input("Approuver ? [o/n] ").strip().lower()
        if answer in {"o", "oui", "y", "yes", "n", "non", "no"}:
            break
    comment = input("Commentaire (facultatif) : ").strip()
    return {
        "decision": "approve" if answer in {"o", "oui", "y", "yes"} else "reject",
        "by": getpass.getuser(),
        "comment": comment,
    }


def print_run(run: Run) -> None:
    s = run.state
    print("\n" + "=" * 60)
    print(f"Run {run.run_id} : {run.status}  {STATUS_TEXT.get(run.status, '')}")
    for name, r in s.get("results", {}).items():
        icon = ICONS.get(r["status"], " ")
        print(f"  {icon} {name:<17} {r['status']:<15} {r['duration_s']}s")
        for reason in r["reasons"]:
            print(f"      → {reason}")
    for d in s.get("decisions", []):
        print(f"  ☐ {d['gate']} : {d['decision']} par {d['by']}")
    q = run.question
    if q:
        print(f"\n  En attente : {q['question']}")
        if q.get("file_to_review"):
            print(f"  Fichier à relire : {q['file_to_review']}")
        print(f"  → python -m orchestrator approve {run.run_id}")
        print(f"  → python -m orchestrator reject  {run.run_id}")
    print(f"Fichiers : {run.run_dir}")


def cmd_run(args) -> Run:
    print(f"Pipeline sur {args.project_dir}… (le validateur peut prendre jusqu'à une minute)")
    run = Run.start(args.project_dir, args.runs_dir)
    if args.auto_approve:
        while not run.finished:
            print(f"── {run.question['gate']} : approuvé automatiquement (--auto-approve)")
            run.answer({"decision": "approve", "by": "auto-approve", "comment": ""})
    elif not args.no_wait:
        while not run.finished:
            run.answer(ask_human(run.question))
    return run


def cmd_status(args) -> None:
    if args.run_id:
        print_run(Run.load(args.run_id, args.runs_dir))
        return
    print(f"Base : {describe_backend(args.runs_dir)}\n")
    runs = list_runs(args.runs_dir)
    if not runs:
        print("Aucun run enregistré.")
        return
    print(f"{'RUN':<14}{'STATUT':<19}{'PROJET':<28}DÉCISIONS")
    for run in runs:
        s = run.state
        project = s.get("project_dir", "?").rstrip("/").split("/")[-1]
        print(f"{run.run_id:<14}{run.status:<19}{project:<28}{len(s.get('decisions', []))}")


def cmd_decide(args, decision: str) -> Run:
    run = Run.load(args.run_id, args.runs_dir)
    run.answer({"decision": decision, "by": getpass.getuser(), "comment": args.message})
    return run


def cmd_resume(args) -> Run:
    run = Run.load(args.run_id, args.runs_dir)
    print(f"Reprise du run {run.run_id} à l'étape : {', '.join(run.snapshot.next)}")
    run.resume()
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m orchestrator")
    parser.add_argument("--runs-dir", default="runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="exécuter le pipeline sur un projet")
    p.add_argument("project_dir")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--auto-approve", action="store_true",
                      help="approuver toutes les gates (tests uniquement)")
    mode.add_argument("--no-wait", action="store_true",
                      help="ne pas poser la question : s'arrêter à la première pause")

    p = sub.add_parser("status", help="lister les runs, ou détailler un run")
    p.add_argument("run_id", nargs="?")

    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} la décision en attente d'un run")
        p.add_argument("run_id")
        p.add_argument("-m", "--message", default="", help="commentaire")

    p = sub.add_parser("resume", help="reprendre un run arrêté par un plantage")
    p.add_argument("run_id")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "status":
            cmd_status(args)
            return 0
        if args.cmd == "run":
            run = cmd_run(args)
        elif args.cmd in ("approve", "reject"):
            run = cmd_decide(args, args.cmd)
        else:
            run = cmd_resume(args)
    except (RunNotFound, RuntimeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print_run(run)
    return 0 if run.status in ("DONE", "WAITING_HUMAN") else 1


if __name__ == "__main__":
    sys.exit(main())
