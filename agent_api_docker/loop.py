"""Pipeline principal : rapport -> generation -> dossier d'execution -> validation -> tests.

Si la validation ou les tests echouent, le LLM propose des patchs (fixer.py), le code les applique sur les
fichiers generes puis on revalide, jusqu'a MAX_ITERATIONS corrections.
Avec docker=True : build de l'image, conteneur, /health, tests contre le conteneur (memes corrections LLM), puis
avec push=True publication sur Docker Hub apres approbation (docker_stage.py).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config
from .api_tests import TestRunResult, prepare_run_dir, run_api_tests
from .discovery_agent import DiscoveryResult, find_report
from .docker_stage import ApprovalRequest, DockerResult, PushResult, missing_push_settings, publish_image, run_docker_stage
from .fixer import FixAttempt, FixError, apply_patches, build_context, classify_failure, propose_fix, short_failure
from .generator import generate_all
from .llm import LLMClient, LLMError, make_client
from .report_loader import ReportError, load_report, resolve_report_path
from .tools.docker import DockerCLI
from .tools.files import write_file
from .traces import log_step
from .validate_api import ValidationResult, validate_api

TOTAL_STEPS = 5

CHECK_LABELS = {
    "file_exists": "api.py present",
    "syntax": "Syntaxe Python valide",
    "lint": "Lint (ruff)",
    "import_app": "Import de l'application FastAPI",
    "routes": "Routes GET /health et POST /predict",
    "contract_input": "Contrat d'entree = rapport d'analyse",
    "contract_output": "Contrat de sortie = rapport d'analyse",
}


@dataclass
class PipelineResult:
    project_id: str
    workspace: Path
    run_dir: Path
    files: list[str]
    validation: ValidationResult
    tests: Optional[TestRunResult]
    duration_s: float
    fixes: list[FixAttempt] = field(default_factory=list)
    docker: Optional[DockerResult] = None      # renseigne si l'etape Docker a ete demandee ET tentee
    push: Optional[PushResult] = None          # renseigne si le push a ete demande

    @property
    def api_ok(self) -> bool:
        return self.validation.ok and self.tests is not None and self.tests.ok

    @property
    def ok(self) -> bool:
        """API valide ; et, si demandes : image Docker validee, push sans erreur (un push refuse par l'utilisateur n'est pas une erreur)."""
        return (self.api_ok and (self.docker is None or self.docker.ok)
                and (self.push is None or self.push.error is None))

    def to_dict(self) -> dict:
        t = self.tests
        return {
            "project_id": self.project_id, "ok": self.ok, "files": self.files,
            "validation": self.validation.to_dict(),
            "tests": None if t is None else {"started": t.started, "passed": t.passed, "failed": t.failed,
                                             "skipped": t.skipped, "error_category": t.error_category,
                                             "failures": [x for x in t.tests if x["status"] == "failed"]},
            "fixes": [f.to_dict() for f in self.fixes],
            "docker": None if self.docker is None else self.docker.to_dict(),
            "push": None if self.push is None else self.push.to_dict(),
            "duration_s": self.duration_s,
        }


def _fmt_event(ev: dict) -> str:
    """Une ligne lisible par evenement de l'agent de recherche."""
    k = ev["kind"]
    if k == "thought":
        return "agent : " + " ".join(ev["text"].split())[:300]
    if k == "tool":
        args = ", ".join(f"{a}={v!r}" for a, v in ev["args"].items())
        return f"-> {ev['name']}({args})"
    if k == "result":
        lines = [ln.strip() for ln in ev["text"].splitlines() if ln.strip()][:3]
        return "   = " + " | ".join(ln[:120] for ln in lines)
    if k == "done":
        return f"retenu : rapport {ev['report']} | projet {ev['project_dir']}"
    if k == "give_up":
        return "abandon : " + ev["reason"][:200]
    return str(ev)


def _discover(project_id: str, llm: Optional[LLMClient], on_detail: Callable[[str], None],
              events: list[dict], forced: bool = False) -> DiscoveryResult:
    """Lance l'agent LLM de recherche ; toute panne LLM devient une ReportError lisible."""
    on_detail("l'agent LLM explore le depot pour retrouver le rapport"
              + ("" if forced else " (absent des emplacements habituels)"))

    def on_event(ev: dict) -> None:
        events.append(ev)
        on_detail(_fmt_event(ev))

    try:
        return find_report(project_id, llm=llm, max_steps=config.DISCOVERY_MAX_STEPS, on_event=on_event)
    except LLMError as e:
        raise ReportError(f"Rapport introuvable et agent de recherche indisponible : {e}") from e


def run(project_id: str, report_path: str | None = None, workspace: Path | None = None) -> Path:
    """Generation seule (sans validation ni tests)."""
    report = load_report(project_id, report_path)
    ws = workspace or (config.WORKSPACES_DIR / project_id / "generated")
    ws.mkdir(parents=True, exist_ok=True)
    files = generate_all(report)
    for rel, content in files.items():
        write_file(ws, rel, content)
    log_step(ws, step="generate", files=list(files))
    return ws


def _evaluate(report, run_dir: Path, ws: Path, python_exe: Optional[str], on_detail: Callable[[str], None],
              first: bool, step: Callable[[str], None]) -> tuple[ValidationResult, Optional[TestRunResult]]:
    """Validation statique puis tests runtime (une passe). Les tests ne sont lances que si la validation passe."""
    validation = validate_api(report, run_dir, python_exe=python_exe)
    log_step(ws, step="validate", **validation.to_dict())
    for c in validation.checks:
        status = "IGNORE" if c.skipped else ("OK" if c.passed else "ECHEC")
        line = f"[{status:<6}] {CHECK_LABELS.get(c.name, c.name)}"
        if not c.passed and c.detail:
            line += " : " + " ".join(c.detail.split())[:200]
        on_detail(line)

    if first:
        step("Tests de l'API (serveur local + tests generes depuis le rapport)")
    tests = None
    if not validation.ok:
        on_detail("tests non lances : la validation a echoue (voir ci-dessus)")
        return validation, None
    on_detail("demarrage d'uvicorn sur un port libre, attente de /health, puis pytest")
    tests = run_api_tests(report, run_dir, python_exe=python_exe)
    log_step(ws, step="tests", passed=tests.passed, failed=tests.failed, started=tests.started)
    if not tests.started:
        on_detail(f"le serveur n'a pas demarre (erreur {tests.error_category})")
    else:
        on_detail(f"{tests.passed} passes, {tests.failed} echecs, {tests.skipped} ignores ({tests.duration_s} s)")
        by_name = {x["name"]: x["status"] for x in tests.tests}
        if by_name.get("test_api_matches_direct_model_call") == "passed":
            on_detail("coherence verifiee : l'API donne les memes predictions que le modele appele directement")
        for x in tests.tests:
            if x["status"] == "failed":
                on_detail(f"[ECHEC ] {x['name']} : {x['message'][:150]}")
    return validation, tests


def run_pipeline(project_id: str, report_path: str | None = None, project_dir: Path | None = None,
                 workspace: Path | None = None, python_exe: str | None = None,
                 on_step: Callable[[str], None] = lambda msg: None,
                 on_detail: Callable[[str], None] = lambda msg: None,
                 agent: str = "auto", llm: Optional[LLMClient] = None,
                 fix: bool = True, max_fixes: Optional[int] = None,
                 docker: bool = False, push: bool = False, max_docker_fixes: Optional[int] = None,
                 approve_push: Optional[Callable[[ApprovalRequest], bool]] = None,
                 docker_cli: Optional[DockerCLI] = None, namespace: Optional[str] = None) -> PipelineResult:
    """Genere l'API, la valide, puis la teste.

    on_step   : appele au DEBUT de chaque etape ("[2/5] Generation ...")
    on_detail : appele pour chaque information utile de l'etape en cours
    agent     : "auto"   = chemins habituels d'abord, agent LLM de recherche si rien n'est trouve
                "always" = l'agent LLM cherche toujours (sauf --report explicite)
                "never"  = chemins habituels uniquement
    fix       : True = si validation/tests echouent, le LLM corrige les fichiers generes puis on reteste
    max_fixes : nombre max de corrections (defaut : config.MAX_ITERATIONS)
    docker    : True = apres une API valide, build de l'image + conteneur + tests contre le conteneur (+ corrections LLM)
    push      : True = publie l'image validee sur Docker Hub (implique docker) ; REQUIERT approve_push
    approve_push : fonction (ApprovalRequest) -> bool, appelee par le CODE avant tout push (jamais par le LLM) ;
                   absente = pas de push
    max_docker_fixes : budget de corrections pendant l'etape Docker (defaut : config.MAX_DOCKER_ITERATIONS)
    """
    docker = docker or push
    if push and (missing := missing_push_settings(namespace)):     # echec immediat, avant tout build
        raise RuntimeError("Push Docker Hub impossible : a renseigner dans " + str(config.REPO_ROOT / ".env")
                           + " -> " + ", ".join(missing) + "  (modele : .env.example)")
    t0 = time.time()
    counter = {"n": 0}
    total = TOTAL_STEPS + (2 if docker else 0) + (1 if push else 0)

    def step(title: str) -> None:
        counter["n"] += 1
        on_step(f"[{counter['n']}/{total}] {title}")

    # 1 ---------------------------------------------------------------- rapport
    step("Localisation, lecture et validation du rapport d'analyse")
    events: list[dict] = []
    if agent not in ("auto", "always", "never"):
        raise ValueError(f"agent doit etre auto|always|never, recu : {agent!r}")
    if agent == "always" and not report_path:
        found = _discover(project_id, llm, on_detail, events, forced=True)
        report_path, project_dir = str(found.report_path), project_dir or found.project_dir
    else:
        try:
            resolve_report_path(project_id, report_path)
        except ReportError:
            if agent == "never":
                raise
            found = _discover(project_id, llm, on_detail, events)
            report_path, project_dir = str(found.report_path), project_dir or found.project_dir
    on_detail(f"rapport     : {os.path.relpath(resolve_report_path(project_id, report_path))}")
    report = load_report(project_id, report_path)
    ep = report.inference.entrypoint
    on_detail(f"projet     : {report.project_name}")
    on_detail(f"framework  : {report.ml.framework} {report.ml.framework_version} | python {report.python.version}"
              f" | entree {report.io.input_type}")
    on_detail(f"modele     : {report.ml.model_file} ({'present' if report.ml.model_exists_in_project else 'a entrainer'})")
    on_detail(f"inference  : {ep.file} -> {ep.function}() | {len(report.io.input_schema)} champs d'entree")
    for u in report.unresolved:
        on_detail(f"point ouvert : {u.question}")

    # 2 ------------------------------------------------------------- generation
    step("Generation des fichiers a partir des templates")
    ws = Path(workspace or (config.WORKSPACES_DIR / project_id))
    if ws.exists():
        shutil.rmtree(ws)          # chaque execution repart de zero
    generated = ws / "generated"
    generated.mkdir(parents=True)
    files = generate_all(report)
    for rel, content in files.items():
        write_file(generated, rel, content)
        on_detail(f"{rel} ({len(content.splitlines())} lignes)")
    log_step(ws, step="generate", files=list(files))
    if events:
        log_step(ws, step="discover", events=events)   # trajectoire de l'agent = futur dataset de fine-tuning

    # 3 ------------------------------------------------------ dossier d'execution
    step("Preparation du dossier d'execution (code du projet + fichiers generes)")
    src = Path(project_dir or (config.EVAL_PROJECTS_DIR / project_id))
    if not src.is_dir():
        raise FileNotFoundError(f"Dossier du projet introuvable : {src}  (utiliser --project-dir ou --agent)")
    if (src / report.ml.model_file).exists():
        on_detail("modele deja present : copie telle quelle")
    else:
        cmd = report.ml.model_generation.command if report.ml.model_generation else "?"
        on_detail(f"modele absent : entrainement avec `{cmd}` (quelques secondes)")
    t = time.time()
    run_dir = prepare_run_dir(src, generated, report, ws / "run", python_exe=python_exe)
    on_detail(f"dossier pret en {round(time.time() - t, 1)} s : {run_dir}")

    # 4-5 ------------------------------------------- validation + tests (+ corrections LLM)
    max_fixes = config.MAX_ITERATIONS if max_fixes is None else max_fixes
    fixes: list[FixAttempt] = []
    tried: set[str] = set()
    fix_llm = llm
    validation: ValidationResult
    tests: Optional[TestRunResult] = None
    attempt = 0
    while True:
        attempt += 1
        if attempt == 1:
            step("Validation statique de api.py")
        else:
            on_step(f"[correction {attempt - 1}/{max_fixes}] Nouvelle validation + nouveaux tests apres correction")
        validation, tests = _evaluate(report, run_dir, ws, python_exe, on_detail, first=(attempt == 1), step=step)
        if validation.ok and tests is not None and tests.ok:
            break
        if not fix or len(fixes) >= max_fixes:
            if fix:
                on_detail(f"budget de corrections epuise ({max_fixes})")
            break

        # --- echec : le LLM propose une correction, le code la controle et l'applique
        if not validation.ok:
            failure, hint = validation.summary(), validation.failures[0].category
        else:
            failure, hint = tests.summary(), tests.error_category
            if tests.started and tests.server_logs:
                failure += "\n--- logs serveur ---\n" + short_failure(tests.server_logs, 40)
        failure = short_failure(failure) if len(failure) > 4000 else failure
        category = classify_failure(failure, hint)
        n = len(fixes) + 1
        on_detail(f"correction automatique {n}/{max_fixes} : categorie {category}, le LLM analyse l'echec")
        att = FixAttempt(iteration=n, category=category, failure=failure)
        try:
            fix_llm = fix_llm or make_client()
            proposal = propose_fix(fix_llm, build_context(report, generated, run_dir, category, failure, fixes))
        except (LLMError, FixError) as e:
            on_detail(f"correction impossible : {e}")
            att.errors.append(str(e))
            fixes.append(att)
            log_step(ws, step="fix", **att.to_dict())
            break
        att.diagnosis, att.target, att.model = proposal.diagnosis, proposal.target, proposal.model
        att.patches = [{"path": p.path, "old": p.old, "new": p.new} for p in proposal.patches]
        on_detail("diagnostic : " + " ".join(proposal.diagnosis.split())[:300])
        att.applied, att.errors = apply_patches(generated, proposal.patches, tried)
        for e in att.errors:
            on_detail("patch refuse : " + e[:200])
        fixes.append(att)
        log_step(ws, step="fix", **att.to_dict())     # trajectoire = futur dataset de fine-tuning
        if not att.applied:
            on_detail("aucun patch applique : arret des corrections")
            break
        on_detail(f"{att.applied} patch(s) applique(s) sur : " + ", ".join(sorted({p.path for p in proposal.patches})))
        for f in generated.iterdir():                 # republie les fichiers generes dans le dossier d'execution
            if f.is_file():
                shutil.copy2(f, run_dir / f.name)
        if any(p.path in ("requirements.api.txt", "Dockerfile") for p in proposal.patches):
            on_detail("note : dependances/Dockerfile modifies, ils ne seront verifies qu'au build Docker (pas en local)")

    # 6-8 ---------------------------------------------- image Docker (+ corrections LLM) puis push approuve
    docker_res: Optional[DockerResult] = None
    push_res: Optional[PushResult] = None
    if docker and not (validation.ok and tests is not None and tests.ok):
        on_detail("etape Docker ignoree : l'API locale doit d'abord passer la validation et les tests")
    elif docker:
        def _reeval() -> Optional[tuple[str, str]]:
            nonlocal validation, tests
            validation, tests = _evaluate(report, run_dir, ws, python_exe, on_detail, first=False, step=step)
            if validation.ok and tests is not None and tests.ok:
                return None
            if not validation.ok:
                return validation.summary(), validation.failures[0].category
            return tests.summary(), tests.error_category or classify_failure(tests.summary())

        docker_res = run_docker_stage(report, generated, run_dir, ws, step=step, on_step=on_step, on_detail=on_detail,
                                      llm=fix_llm, fixes=fixes, tried=tried, max_fixes=max_docker_fixes, fix=fix,
                                      reeval=_reeval, docker=docker_cli, python_exe=python_exe)
        if push and docker_res.ok:
            step("Publication sur Docker Hub (apres approbation)")
            push_res = publish_image(docker_res, report, approve_push, on_detail, ws, docker=docker_cli, namespace=namespace)
        elif push:
            on_detail("push ignore : l'image n'a pas passe toutes les verifications")

    result = PipelineResult(project_id, ws, run_dir, list(files), validation, tests, round(time.time() - t0, 2), fixes,
                            docker_res, push_res)
    (ws / "result.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result
