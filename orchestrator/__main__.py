"""
Ligne de commande de l'orchestrateur.

    python -m orchestrator run eval/projects/credit_risk            # interactif
    python -m orchestrator run eval/projects/credit_risk --no-wait  # s'arrête à la 1re pause
    python -m orchestrator status                                   # liste des runs
    python -m orchestrator status <run_id>                          # détail d'un run
    python -m orchestrator approve <run_id> [-m "commentaire"]
    python -m orchestrator reject  <run_id> [-m "commentaire"]
    python -m orchestrator resume  <run_id>                         # après un plantage
    python -m orchestrator monitor <run_id>                         # Agent 4 : réentraîner si demandé
    python -m orchestrator slack                                    # validations humaines dans Slack
    python -m orchestrator run eval/projects/credit_risk --slack    # le run attend les décisions Slack
    python -m orchestrator export-decisions -o decisions.jsonl      # décisions du superviseur
    python -m orchestrator check-agent agent2 mlflow_agent.orchestrator_adapter:run  # avant intégration

Superviseur (qui décide de la suite après chaque étape) :
    python -m orchestrator run eval/projects/credit_risk --supervisor gemini
    python -m orchestrator run eval/projects/credit_risk --supervisor gemini --simulate-failure transient

L'état de chaque run est sauvegardé dans runs/checkpoints.sqlite :
un run en pause peut être approuvé plus tard, depuis un autre terminal.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

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
    if question.get("failed_step"):
        print(f"Étape en échec : {question['failed_step']}")
        for reason in question.get("reasons", [])[:3]:
            print(f"  → {reason}")
    if question.get("supervisor_explanation"):
        print(f"Avis du superviseur : {question['supervisor_explanation']}")
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
    if s.get("supervisor_mode", "rules") != "rules":
        print(f"\n  Superviseur : {s['supervisor_mode']}")
        for d in s.get("supervision", []):
            note = f"  [secours : {d['fallback']}]" if d.get("fallback") else ""
            print(f"    après {d['after']:<16} → {d['action']:<16} {d['explanation']}{note}")
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
    if not Path(args.project_dir).is_dir():
        raise RuntimeError(f"{args.project_dir} n'est pas un dossier (chemin relatif au dossier courant : "
                           f"{Path.cwd()})")
    from .graph import AGENT_ORIGIN
    print(f"Pipeline sur {args.project_dir}… (le validateur peut prendre jusqu'à une minute)")
    print("Agents : " + " · ".join(f"{name} {origin}" for name, origin in AGENT_ORIGIN.items()
                                   if name != "propose_manifest"))
    from .llm import default_mode
    print(f"Superviseur : {args.supervisor or default_mode()}")
    if args.simulate_drift and AGENT_ORIGIN.get("agent4") == "réel":
        print("Note : --simulate-drift ne concerne que le stub de l'Agent 4 ; le vrai agent décide sur le "
              "trafic observé. Pour la démo du drift : ORCH_AGENTS=stub python -m orchestrator run ...")
    if args.simulate_failure and AGENT_ORIGIN.get("agent3") == "réel":
        print("Note : --simulate-failure ne concerne que le stub de l'Agent 3.")
    # Un humain répond ici : les agents peuvent poser leurs propres questions (ex. compte Docker Hub de l'Agent 3).
    if args.slack:
        args.no_wait = True
        print("Décisions dans Slack : le service doit tourner (python -m orchestrator slack).")
    if not args.auto_approve and not args.no_wait and sys.stdin.isatty():
        os.environ["ORCH_INTERACTIVE"] = "1"
    simulate = None
    if args.simulate_failure or args.simulate_drift:
        simulate = {"failure": args.simulate_failure, "drift": args.simulate_drift}
    run = Run.start(args.project_dir, args.runs_dir, supervisor=args.supervisor, simulate=simulate)
    if args.auto_approve:
        while not run.finished:
            print(f"── {run.question['gate']} : approuvé automatiquement (--auto-approve)")
            run.answer({"decision": "approve", "by": "auto-approve", "comment": ""})
    elif not args.no_wait:
        while not run.finished:
            run.answer(ask_human(run.question))
    return run


def cmd_monitor(args) -> Run:
    """Relance l'Agent 4 sur un run déployé ; une demande de réentraînement repasse par les gates."""
    run = Run.load(args.run_id, args.runs_dir)
    print(f"Monitoring du run {run.run_id} : l'Agent 4 relit la production…")
    if not args.no_wait and sys.stdin.isatty():
        os.environ["ORCH_INTERACTIVE"] = "1"
    run.monitor()
    if not args.no_wait:
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


def cmd_export(args) -> None:
    """Exporte les décisions du superviseur (une par ligne, format JSONL)."""
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for run in list_runs(args.runs_dir):
            for d in run.state.get("supervision", []):
                if "context" not in d:  # décision ancienne, sans la situation enregistrée
                    continue
                if args.supervisor and d["decider"] != args.supervisor:
                    continue
                row = {"run_id": run.run_id, **d}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"{count} décisions exportées dans {args.output}")


def cmd_decide(args, decision: str) -> Run:
    if sys.stdin.isatty():
        os.environ["ORCH_INTERACTIVE"] = "1"
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
    p.add_argument("--supervisor", choices=["rules", "gemini"], default=None,
                   help="qui décide de la suite (défaut : ORCH_SUPERVISOR, sinon gemini si "
                        "GEMINI_API_KEY est définie, sinon rules)")
    p.add_argument("--simulate-failure", choices=["transient", "permanent"], default=None,
                   help="démo : l'Agent 3 échoue (transient : une fois ; permanent : toujours)")
    p.add_argument("--simulate-drift", action="store_true", help="démo : l'Agent 4 détecte un drift")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--auto-approve", action="store_true",
                      help="approuver toutes les gates (tests uniquement)")
    mode.add_argument("--no-wait", action="store_true",
                      help="ne pas poser la question : s'arrêter à la première pause")
    mode.add_argument("--slack", action="store_true",
                      help="les décisions se prennent dans Slack (service : python -m orchestrator slack)")

    p = sub.add_parser("status", help="lister les runs, ou détailler un run")
    p.add_argument("run_id", nargs="?")

    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} la décision en attente d'un run")
        p.add_argument("run_id")
        p.add_argument("-m", "--message", default="", help="commentaire")

    p = sub.add_parser("resume", help="reprendre un run arrêté par un plantage")
    p.add_argument("run_id")

    p = sub.add_parser("monitor", help="relancer l'Agent 4 sur un run déployé (réentraînement si demandé)")
    p.add_argument("run_id")
    p.add_argument("--no-wait", "--slack", dest="no_wait", action="store_true",
                   help="ne pas poser la question : s'arrêter à la gate (décision dans Slack ou approve/reject)")

    sub.add_parser("slack", help="service Slack : validations humaines, réentraînements de l'Agent 4")

    p = sub.add_parser("export-decisions", help="exporter les décisions du superviseur (JSONL)")
    p.add_argument("-o", "--output", default="decisions.jsonl", help="fichier de sortie")
    p.add_argument("--supervisor", choices=["rules", "gemini"], default=None,
                   help="garder seulement les décisions de ce décideur")

    p = sub.add_parser("check-agent", help="vérifier qu'un agent respecte le contrat avant de l'intégrer")
    p.add_argument("name", help="agent1, propose_manifest, agent2, agent3, agent4 ou repair")
    p.add_argument("target", help="fonction de l'agent, ex. mlflow_agent.orchestrator_adapter:run")
    p.add_argument("--project", default=None, help="projet de test (défaut : selon l'agent)")

    args = parser.parse_args(argv)
    if args.cmd == "slack":
        from .slack import serve
        return serve(args.runs_dir)
    if args.cmd == "check-agent":
        from .check_agent import main as check_main
        return check_main(args.name, args.target, args.project, str(Path(args.runs_dir) / "check"))
    try:
        if args.cmd == "status":
            cmd_status(args)
            return 0
        if args.cmd == "export-decisions":
            cmd_export(args)
            return 0
        if args.cmd == "run":
            run = cmd_run(args)
        elif args.cmd == "monitor":
            run = cmd_monitor(args)
        elif args.cmd in ("approve", "reject"):
            run = cmd_decide(args, args.cmd)
        else:
            run = cmd_resume(args)
    except (RunNotFound, RuntimeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print_run(run)
    return 0 if run.status in ("DONE", "WAITING_HUMAN", "CHAMPION_KEPT") else 1


if __name__ == "__main__":
    sys.exit(main())
