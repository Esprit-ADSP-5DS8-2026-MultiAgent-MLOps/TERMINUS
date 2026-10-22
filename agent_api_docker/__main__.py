"""CLI : python -m agent_api_docker <project_id> [--agent | --no-agent] [--serve] [--show]"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from .loop import run, run_pipeline
from .report_loader import NeedsUserInput, ReportError

p = argparse.ArgumentParser(prog="agent_api_docker", description="Genere, valide et teste l'API d'un projet ML.")
p.add_argument("project_id", help="ex: credit_risk (dossier eval/projects/<id>)")
p.add_argument("--report", help="chemin explicite du rapport d'analyse JSON")
p.add_argument("--project-dir", help="dossier du code source (defaut: eval/projects/<id>)")
mode = p.add_mutually_exclusive_group()
mode.add_argument("--agent", action="store_true", help="l'agent LLM cherche toujours le rapport (sinon : seulement si les chemins habituels echouent)")
mode.add_argument("--no-agent", action="store_true", help="n'utilise jamais le LLM : chemins habituels uniquement")
p.add_argument("--no-fix", action="store_true", help="desactive la correction automatique par le LLM (echec = on s'arrete)")
p.add_argument("--max-fixes", type=int, help="nombre max de corrections LLM (defaut : AGENT_MAX_ITERATIONS = 6)")
p.add_argument("--docker", action="store_true", help="apres une API valide : build de l'image, conteneur, tests contre le conteneur (corrections LLM comprises)")
p.add_argument("--verbose", "-v", action="store_true", help="affiche toutes les lignes du build Docker (sinon : etapes + durees ; log complet dans workspaces/<id>/docker_build.log)")
p.add_argument("--push", action="store_true", help="publie l'image validee sur Docker Hub apres approbation (implique --docker)")
p.add_argument("--yes", action="store_true", help="approuve le push sans question (l'approbation reste liee au sha256 de l'image testee)")
p.add_argument("--namespace", help="espace de nom Docker Hub (defaut : DOCKER_IMAGE_NAMESPACE ou DOCKERHUB_USERNAME)")
p.add_argument("--max-docker-fixes", type=int, help="nombre max de corrections LLM pendant l'etape Docker (defaut : AGENT_MAX_DOCKER_ITERATIONS = 6)")
p.add_argument("--generate-only", action="store_true", help="ecrit les fichiers sans valider ni tester")
p.add_argument("--show", action="store_true", help="affiche api.py et le Dockerfile generes a la fin")
p.add_argument("--serve", action="store_true", help="apres un succes, lance l'API pour l'essayer (/docs)")
p.add_argument("--port", type=int, default=8000)
args = p.parse_args()
if args.verbose:
    from . import config
    config.DOCKER_VERBOSE = True


def rel(path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return str(path)


def on_step(msg: str) -> None:
    print("\n" + msg, flush=True)


def on_detail(msg: str) -> None:
    print("      " + msg, flush=True)


def approve_push(req) -> bool:
    """Approbation humaine, en CODE (jamais par le LLM). Liee au sha256 de l'image qui vient de passer les tests."""
    print(f"\nImage validee : {req.image_id[:19]}...  ({req.size_mb} Mo)")
    print(f"Destination   : docker.io/{req.repository}  (tags : {', '.join(req.tags)})")
    print("Attention : un depot Docker Hub cree par ce push est PUBLIC par defaut (modele inclus). "
          "S'il est confidentiel, creer d'abord le depot en prive sur hub.docker.com.")
    if args.yes:
        print("--yes : push approuve pour ce sha256.")
        return True
    if not sys.stdin.isatty():
        print("Entree non interactive : push non approuve (utiliser --yes pour l'autoriser).")
        return False
    return input("Pousser cette image sur Docker Hub ? [o/N] ").strip().lower() in ("o", "oui", "y", "yes")


try:
    if args.generate_only:
        print("Fichiers generes dans :", run(args.project_id, args.report))
        sys.exit(0)

    print(f"=== Agent API/Docker : {args.project_id} ===")
    result = run_pipeline(args.project_id, args.report,
                          Path(args.project_dir) if args.project_dir else None,
                          on_step=on_step, on_detail=on_detail,
                          agent="always" if args.agent else "never" if args.no_agent else "auto",
                          fix=not args.no_fix, max_fixes=args.max_fixes,
                          docker=args.docker or args.push, push=args.push, approve_push=approve_push,
                          max_docker_fixes=args.max_docker_fixes, namespace=args.namespace)

    print("\n" + "=" * 60)
    print("RESULTAT :", "SUCCES" if result.ok else "ECHEC", f"({result.duration_s} s)")
    print("Fichiers generes :", rel(result.workspace / "generated"))
    print("Details          :", rel(result.workspace / "result.json"))
    if result.fixes:
        done = [f for f in result.fixes if f.applied]
        print(f"Corrections LLM  : {len(done)} appliquee(s) sur {len(result.fixes)} tentative(s)")
        for f in result.fixes:
            print(f"   {f.iteration}. [{f.category}] {' '.join(f.diagnosis.split())[:160] or 'pas de diagnostic'}")
    if result.docker:
        dk = result.docker
        print("Image Docker     :", f"{dk.image} ({dk.size_mb} Mo, {dk.image_id[:19]}...)" if dk.ok else f"ECHEC a l'etape {dk.stage} ({dk.category})")
        if not dk.ok:
            print("\nCause (Docker) :\n" + dk.failure[-1500:])
    if result.push:
        pr = result.push
        if pr.pushed:
            print("Docker Hub       :", f"docker.io/{pr.repository} tags {', '.join(pr.tags)} digest {pr.digest}")
        elif pr.declined:
            print("Docker Hub       : push non approuve, image locale conservee")
        else:
            print("Docker Hub       : ECHEC :", pr.error)
    if result.tests and not result.tests.ok:
        print("\nCause :\n" + result.tests.summary())
    if not result.api_ok and not result.validation.ok:
        print("\nCause :\n" + result.validation.summary())
    if result.ok:
        print(f"Relancer l'API   : cd {rel(result.run_dir)} && python -m uvicorn api:app --port {args.port}")

    if args.show:
        for name in ("api.py", "Dockerfile"):
            print(f"\n----- {name} " + "-" * 40)
            print((result.workspace / "generated" / name).read_text(encoding="utf-8"))

    if result.ok and args.serve:
        print(f"\nAPI en ligne : http://127.0.0.1:{args.port}/docs   (Ctrl+C pour arreter)")
        subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(args.port)],
                       cwd=result.run_dir)
    sys.exit(0 if result.ok else 1)
except NeedsUserInput as e:
    print("Questions pour l'utilisateur :", *e.questions, sep="\n - ")
    sys.exit(2)
except (ReportError, RuntimeError, FileNotFoundError) as e:
    print("Erreur :", e)
    sys.exit(2)
