"""Etape Docker : build -> conteneur -> /health -> tests contre le conteneur -> (push approuve sur Docker Hub).

Meme principe que le reste de l'agent :
  - le CODE execute Docker, mesure, controle et decide (budgets, approbation, fichiers modifiables) ;
  - le LLM (fixer.py) ne fait que diagnostiquer les echecs et proposer des patchs sur les fichiers GENERES
    (Dockerfile, requirements.api.txt, api.py, .dockerignore, docker-compose.yml).
Les problemes d'environnement (daemon arrete, disque plein, quota Docker Hub, timeout) ne se corrigent pas par un patch :
on s'arrete avec une explication claire au lieu de gaspiller le budget de corrections.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config
from .api_tests import TestRunResult, run_tests_against_url
from .errors import classify
from .fixer import FixAttempt, FixError, apply_patches, build_context, classify_failure, propose_fix, short_failure
from .llm import LLMClient, LLMError, make_client
from .schemas import AnalysisReport
from .tools.checks import healthcheck, lint_dockerfile, missing_copy_sources
from .tools.docker import DockerCLI, DockerError
from .tools.files import sync_files
from .tools.publish import ApprovalRequired, LoginError, PushError, docker_push
from .traces import log_step

# Echecs qu'aucun patch de fichier genere ne peut corriger.
_ENV_PATTERNS = [
    ("REGISTRY_RATE_LIMIT", r"toomanyrequests|pull rate limit"),
    ("DISK_FULL", r"no space left on device"),
    ("DOCKERFILE_UNREADABLE", r"failed to read dockerfile"),      # Docker ne trouve pas le Dockerfile lui-meme (pas un COPY)
]
ENV_CATEGORIES = {"DOCKER_DAEMON", "REGISTRY_RATE_LIMIT", "DISK_FULL", "BUILD_TIMEOUT", "DOCKERFILE_UNREADABLE"}
ENV_ADVICE = {
    "DOCKER_DAEMON": "demarrer Docker Desktop (ou le service docker) puis relancer",
    "REGISTRY_RATE_LIMIT": "limite de telechargement Docker Hub : `docker login` puis relancer, ou attendre",
    "DISK_FULL": "liberer de l'espace (`docker system prune`) puis relancer",
    "DOCKERFILE_UNREADABLE": "Dockerfile absent ou vide dans le dossier de build : probleme de l'agent, pas du contenu du Dockerfile "
                             "(verifier workspaces/<projet>/generated/Dockerfile)",
    "BUILD_TIMEOUT": "build trop long : augmenter AGENT_BUILD_TIMEOUT_S (entrainement du modele inclus dans le build)",
}
# Erreurs reseau passageres pendant le build : on reessaie sans consommer de correction LLM.
_TRANSIENT = re.compile(r"TLS handshake timeout|no such host|Temporary failure in name resolution|i/o timeout|"
                        r"connection reset by peer|unexpected EOF|failed to do request|net/http: request canceled|"
                        r"503 Service Unavailable", re.I)
_BUILD_STEP = re.compile(r"^#\d+ \[.*\]")
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache"}


# --------------------------------------------------------------------------- resultats

@dataclass
class DockerResult:
    ok: bool = False
    stage: str = ""                 # derniere etape atteinte : preflight | local | lint | build | run | health | tests | done
    category: Optional[str] = None
    failure: str = ""
    image: Optional[str] = None     # reference locale (nom:tag)
    image_id: Optional[str] = None  # sha256:... (sert a l'approbation du push)
    size_mb: float = 0.0
    attempts: int = 0
    tests: Optional[TestRunResult] = None
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        t = self.tests
        return {"ok": self.ok, "stage": self.stage, "category": self.category, "failure": self.failure[:2000],
                "image": self.image, "image_id": self.image_id, "size_mb": self.size_mb, "attempts": self.attempts,
                "tests": None if t is None else {"passed": t.passed, "failed": t.failed, "skipped": t.skipped},
                "duration_s": self.duration_s}


@dataclass
class ApprovalRequest:
    """Ce que l'utilisateur approuve (lie au sha256 de l'image testee)."""
    repository: str
    tags: list[str]
    image_id: str
    size_mb: float


@dataclass
class PushResult:
    pushed: bool = False
    declined: bool = False
    repository: str = ""
    tags: list[str] = field(default_factory=list)
    digest: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"pushed": self.pushed, "declined": self.declined, "repository": self.repository,
                "tags": self.tags, "digest": self.digest, "error": self.error}


@dataclass
class _Attempt:
    ok: bool
    stage: str
    category: Optional[str] = None
    failure: str = ""


# --------------------------------------------------------------------------- utilitaires

def image_slug(project_name: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", project_name.lower()).strip("-._") or "model-api"


def local_ref(report: AnalysisReport) -> str:
    return f"{image_slug(report.project_name)}:agent-build"


def _context_listing(run_dir: Path, limit: int = 150) -> list[str]:
    out: list[str] = []
    for p in sorted(Path(run_dir).rglob("*")):
        rel = p.relative_to(run_dir)
        if any(part in _SKIP_DIRS for part in rel.parts) or not p.is_file():
            continue
        out.append(rel.as_posix())
        if len(out) >= limit:
            out.append("... (liste tronquee)")
            break
    return out


def _categorize(logs: str, hint: Optional[str] = None) -> str:
    for cat, pattern in _ENV_PATTERNS:
        if re.search(pattern, logs, re.I):
            return cat
    cat = classify(logs)[0]
    if cat == "DOCKER_DAEMON":
        return cat
    return hint or cat


# --------------------------------------------------------------------------- une passe lint -> build -> run -> tests

def _attempt(d: DockerCLI, report: AnalysisReport, run_dir: Path, ref: str, res: DockerResult, state: dict,
             step: Callable[[str], None], on_detail: Callable[[str], None], reeval: Optional[Callable],
             python_exe: Optional[str], keep_running: bool, sleep: Callable[[float], None]) -> _Attempt:
    port = report.existing_app.port or 8000

    # 0. api.py a change pendant une correction Docker : on la revalide en local avant de rebuilder
    if state["dirty"] and reeval is not None:
        on_detail("api.py modifie : revalidation locale avant le nouveau build")
        bad = reeval()
        if bad:
            return _Attempt(False, "local", bad[1], bad[0])
        state["dirty"] = False

    # 1a. le Dockerfile doit exister dans le contexte de build ; ses COPY doivent viser des fichiers presents
    df = run_dir / "Dockerfile"
    if not df.is_file() or df.stat().st_size < 20:
        msg = f"Dockerfile absent ou vide dans {run_dir}"
        on_detail("[ECHEC ] " + msg)
        return _Attempt(False, "lint", "DOCKERFILE_UNREADABLE", msg)
    missing = missing_copy_sources(run_dir)
    if missing:
        msg = ("COPY vers des fichiers absents du contexte de build : " + ", ".join(missing)
               + ". Le contexte contient les fichiers du projet + les fichiers generes (liste ci-dessous).")
        on_detail("[ECHEC ] " + msg)
        return _Attempt(False, "lint", "COPY_PATH", msg)

    # 1b. hadolint (si installe)
    lint = lint_dockerfile(str(run_dir / "Dockerfile"))
    if lint["skipped"]:
        if not state.get("lint_noted"):
            on_detail(lint["output"])
            state["lint_noted"] = True
    elif not lint["ok"]:
        on_detail("[ECHEC ] hadolint : " + " ".join(lint["output"].split())[:200])
        return _Attempt(False, "lint", "DOCKERFILE_LINT", lint["output"][-3000:])
    else:
        on_detail("[OK    ] hadolint : aucune erreur")

    # 2. build (les erreurs reseau passageres sont reessayees sans correction)
    log_path = Path(run_dir).parent / "docker_build.log"       # log COMPLET, ecrit en direct : `tail -f` pour suivre
    log_path.write_text("", encoding="utf-8")
    titles: dict[str, str] = {}                                # "#13" -> "[builder 4/6] RUN pip install ..."
    last_print = [time.time()]

    def progress(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        num = line.split(" ", 1)[0]
        if _BUILD_STEP.match(line):
            titles[num] = line.split(" ", 1)[1][:90]
            if not config.DOCKER_VERBOSE:
                on_detail("   " + line[:110])
                last_print[0] = time.time()
                return
        if config.DOCKER_VERBOSE:
            on_detail("   | " + line[:160])
            return
        m = re.match(r"^(#\d+) (DONE ([\d.]+)s|CACHED)", line)
        if m and m.group(1) in titles:                          # fin d'etape : ce qui a pris du temps
            on_detail(f"   {m.group(1)} {'(cache)' if m.group(2) == 'CACHED' else 'termine en ' + m.group(3) + ' s'}"
                      f" : {titles[m.group(1)]}")
            last_print[0] = time.time()
        elif time.time() - last_print[0] > 15 and num in titles:   # etape longue : signe de vie
            on_detail(f"   ... en cours : {titles[num]}")
            last_print[0] = time.time()

    on_detail(f"docker build -t {ref} (dossier : {run_dir}) | log complet : {log_path}")
    t0, build = time.time(), {"exit_code": 1, "logs": ""}
    for retry in range(3):
        build = d.build(str(run_dir), ref, config.BUILD_TIMEOUT_S, on_line=progress)
        if build["exit_code"] == 0 or retry == 2 or not _TRANSIENT.search(build["logs"]):
            break
        wait = 5 * (3 ** retry)
        on_detail(f"erreur reseau pendant le build : nouvel essai dans {wait} s (ne consomme pas de correction)")
        sleep(wait)
    build_s = round(time.time() - t0, 1)
    if build["exit_code"] != 0:
        cat = "BUILD_TIMEOUT" if build["exit_code"] == 124 else _categorize(build["logs"])
        on_detail(f"[ECHEC ] build ({build_s} s) : categorie {cat}")
        return _Attempt(False, "build", cat, short_failure(build["logs"], 60))
    res.image, res.image_id = ref, d.image_id(ref)
    res.size_mb = d.image_size_mb(ref)
    on_detail(f"[OK    ] image construite en {build_s} s : {res.size_mb} Mo ({res.image_id[:19]}...)")

    # 3. conteneur + /health + tests (le conteneur est toujours supprime a la fin)
    if not state.get("announced"):
        step("Conteneur : demarrage, /health, tests de l'API sur l'image")
        state["announced"] = True
    name = f"agent-api-{image_slug(report.project_name)[:30]}-{uuid.uuid4().hex[:8]}"
    run = d.run(ref, port, name=name)
    cid = run.get("container_id")
    try:
        if run["exit_code"] != 0 or not run.get("host_port"):
            logs = run["logs"] + ("\n" + d.logs(cid) if cid else "")
            on_detail("[ECHEC ] le conteneur n'a pas demarre")
            return _Attempt(False, "run", _categorize(logs), short_failure(logs, 60))
        base_url = f"http://127.0.0.1:{run['host_port']}"
        on_detail(f"conteneur {name} sur {base_url} : attente de /health")
        hc = healthcheck(base_url + "/health", config.HEALTHCHECK_TIMEOUT_S,
                         alive=lambda: d.state(cid)["status"] == "running")
        if not hc["ok"]:
            st, logs = d.state(cid), d.logs(cid)
            exited = st["status"] != "running"
            head = (f"Le conteneur s'est arrete (code {st['exit_code']}) avant de repondre." if exited else
                    f"Le conteneur tourne mais /health ne repond pas sur le port {port} ({hc['detail']}). "
                    f"Verifier : uvicorn ecoute sur 0.0.0.0, le port du CMD/EXPOSE = {port} (rapport).")
            failure = head + "\n--- logs du conteneur ---\n" + short_failure(logs, 40)
            cat = _categorize(logs, None if exited else "CONTAINER_UNREACHABLE")
            on_detail(f"[ECHEC ] /health : {head[:150]}")
            return _Attempt(False, "health", cat, failure)
        on_detail(f"[OK    ] /health repond en {hc['elapsed_s']} s")

        tests = run_tests_against_url(report, run_dir, base_url, python_exe=python_exe)
        res.tests = tests
        on_detail(f"{tests.passed} passes, {tests.failed} echecs, {tests.skipped} ignores contre le conteneur")
        if not tests.ok:
            for t in tests.tests:
                if t["status"] == "failed":
                    on_detail(f"[ECHEC ] {t['name']} : {t['message'][:150]}")
            failure = tests.summary() + "\n--- logs du conteneur ---\n" + short_failure(d.logs(cid), 40)
            return _Attempt(False, "tests", classify_failure(failure), failure)
        return _Attempt(True, "done")
    finally:
        if cid and not keep_running:
            d.cleanup(cid)


# --------------------------------------------------------------------------- etape complete (avec corrections)

def run_docker_stage(report: AnalysisReport, generated: Path, run_dir: Path, ws: Path, *,
                     step: Callable[[str], None], on_step: Callable[[str], None], on_detail: Callable[[str], None],
                     llm: Optional[LLMClient] = None, fixes: Optional[list[FixAttempt]] = None,
                     tried: Optional[set[str]] = None, max_fixes: Optional[int] = None, fix: bool = True,
                     reeval: Optional[Callable[[], Optional[tuple[str, str]]]] = None, docker: Optional[DockerCLI] = None,
                     python_exe: Optional[str] = None, keep_running: bool = False,
                     sleep: Callable[[float], None] = time.sleep,
                     applied_out: Optional[list] = None) -> DockerResult:
    """Construit l'image, la lance, la teste ; si ca echoue, le LLM corrige les fichiers generes et on recommence.

    reeval : appelee apres une correction de api.py ; retourne None si la validation+tests locaux passent,
             sinon (echec, categorie).
    applied_out : recoit les patchs effectivement appliques (pour les memoriser si le build + les tests passent).
    """
    d = docker or DockerCLI()
    fixes = [] if fixes is None else fixes
    tried = set() if tried is None else tried
    max_fixes = config.MAX_DOCKER_ITERATIONS if max_fixes is None else max_fixes
    res, t_start = DockerResult(), time.time()

    ok, info = d.available()
    if not ok:
        res.stage, res.category, res.failure = "preflight", "DOCKER_DAEMON", info
        on_detail("[ECHEC ] " + info.splitlines()[0])
        log_step(ws, step="docker", attempt=0, stage="preflight", ok=False, category="DOCKER_DAEMON")
        res.duration_s = round(time.time() - t_start, 1)
        return res
    on_detail(f"Docker disponible (serveur {info})")

    ref, state, used, fix_llm = local_ref(report), {"dirty": False}, 0, llm
    step("Construction de l'image Docker")
    while True:
        res.attempts += 1
        if res.attempts > 1:
            on_step(f"[correction docker {used}/{max_fixes}] Nouveau build + nouveaux tests apres correction")
        att = _attempt(d, report, Path(run_dir), ref, res, state, step, on_detail, reeval, python_exe, keep_running, sleep)
        log_step(ws, step="docker", attempt=res.attempts, stage=att.stage, ok=att.ok, category=att.category,
                 image_id=res.image_id, size_mb=res.size_mb)
        res.stage, res.category, res.failure = att.stage, att.category, att.failure
        if att.ok:
            res.ok, res.category, res.failure = True, None, ""
            break
        if att.category in ENV_CATEGORIES:
            on_detail(f"probleme d'environnement ({att.category}) : aucun patch possible -> {ENV_ADVICE[att.category]}")
            break
        if not fix:
            break
        if used >= max_fixes:
            on_detail(f"budget de corrections Docker epuise ({max_fixes})")
            break

        used += 1
        n = len(fixes) + 1
        on_detail(f"correction automatique Docker {used}/{max_fixes} : etape {att.stage}, categorie {att.category}, le LLM analyse l'echec")
        attempt = FixAttempt(iteration=n, category=att.category or "UNKNOWN", failure=att.failure, stage=att.stage)
        try:
            fix_llm = fix_llm or make_client()
            ctx = build_context(report, generated, Path(run_dir), attempt.category, att.failure, fixes,
                                context_files=_context_listing(Path(run_dir)))
            proposal = propose_fix(fix_llm, ctx)
        except (LLMError, FixError) as e:
            on_detail(f"correction impossible : {e}")
            attempt.errors.append(str(e))
            fixes.append(attempt)
            log_step(ws, step="fix", **attempt.to_dict())
            break
        attempt.diagnosis, attempt.target, attempt.model = proposal.diagnosis, proposal.target, proposal.model
        attempt.patches = [{"path": p.path, "old": p.old, "new": p.new} for p in proposal.patches]
        on_detail("diagnostic : " + " ".join(proposal.diagnosis.split())[:300])
        attempt.applied, attempt.errors = apply_patches(generated, proposal.patches, tried, applied_out)
        for e in attempt.errors:
            on_detail("patch refuse : " + e[:200])
        fixes.append(attempt)
        log_step(ws, step="fix", **attempt.to_dict())
        if not attempt.applied:
            on_detail("aucun patch applique : arret des corrections")
            break
        on_detail(f"{attempt.applied} patch(s) applique(s) sur : " + ", ".join(sorted({p.path for p in proposal.patches})))
        sync_files(Path(generated), Path(run_dir))          # no-op quand generated/ est lui-meme le contexte de build
        state["dirty"] = any(p.path == "api.py" for p in proposal.patches)

    res.duration_s = round(time.time() - t_start, 1)
    return res


# --------------------------------------------------------------------------- publication

def missing_push_settings(namespace: Optional[str] = None) -> list[str]:
    """Reglages Docker Hub absents (a verifier AVANT le build pour ne pas attendre pour rien)."""
    missing = []
    if not config.DOCKERHUB_USERNAME:
        missing.append("DOCKERHUB_USERNAME")
    if not config.DOCKERHUB_TOKEN:
        missing.append("DOCKERHUB_TOKEN")
    if not (namespace or config.DOCKER_IMAGE_NAMESPACE):
        missing.append("DOCKER_IMAGE_NAMESPACE (ou --namespace)")
    return missing


# Fournisseur d'identifiants (saisie dans le terminal, voir terminal.py) : appele avec None la 1re fois, puis avec le
# message d'erreur si Docker Hub refuse la connexion ; retourne (utilisateur, token) ou None pour annuler.
CredentialsProvider = Callable[[Optional[str]], Optional[tuple[str, str]]]
MAX_LOGIN_ATTEMPTS = 3


def publish_image(res: DockerResult, report: AnalysisReport, approve: Optional[Callable[[ApprovalRequest], bool]],
                  on_detail: Callable[[str], None], ws: Path, docker: Optional[DockerCLI] = None,
                  namespace: Optional[str] = None, credentials: Optional[CredentialsProvider] = None) -> PushResult:
    """Pousse l'image testee sur Docker Hub, APRES approbation. Sans callback d'approbation : jamais de push.

    credentials : si fourni, l'utilisateur saisit son nom et son token (dans le terminal) AVANT le push ; ils ne vivent
                  qu'en memoire le temps du push (ni fichier, ni log, ni trace). Sinon : DOCKERHUB_USERNAME / DOCKERHUB_TOKEN de .env.
    """
    out = PushResult()
    user = token = None
    if credentials is not None:
        creds = credentials(None)
        if not creds:
            out.declined = True
            on_detail("push annule : identifiants Docker Hub non fournis, l'image reste locale (" + (res.image or "") + ")")
            log_step(ws, step="push", approved=False, reason="no_credentials")
            return out
        user, token = creds
        # espace de nom : --namespace, sinon DOCKER_IMAGE_NAMESPACE explicite, sinon le compte saisi (jamais celui de .env)
        ns = (namespace or config.DOCKER_IMAGE_NAMESPACE_EXPLICIT or user).strip().lower()
    else:
        ns = (namespace or config.DOCKER_IMAGE_NAMESPACE or "").strip().lower()
    if not ns:
        out.error = ("Espace de nom Docker Hub inconnu : renseigner DOCKERHUB_USERNAME (ou DOCKER_IMAGE_NAMESPACE) "
                     "dans .env, ou passer --namespace.")
        on_detail("[ECHEC ] " + out.error)
        return out
    out.repository = f"{ns}/{image_slug(report.project_name)}"
    out.tags = [time.strftime("%Y%m%d") + "-" + res.image_id.split(":")[-1][:8], "latest"]
    req = ApprovalRequest(out.repository, out.tags, res.image_id, res.size_mb)
    approved = bool(approve and approve(req))
    if not approved:
        out.declined = True
        on_detail("push non approuve : l'image reste locale (" + (res.image or "") + ")")
        log_step(ws, step="push", approved=False, repository=out.repository)
        return out
    attempts = MAX_LOGIN_ATTEMPTS if credentials is not None else 1
    r: dict = {}
    for attempt in range(attempts):
        try:
            r = docker_push(out.repository, out.tags, req.image_id, res.image_id, source_ref=res.image, docker=docker,
                            username=user, token=token, on_line=lambda s: None)
            break
        except LoginError as e:                       # identifiants refuses : on laisse ressaisir (token seulement)
            out.error = str(e)
            on_detail("[ECHEC ] " + out.error)
            creds = credentials(out.error) if credentials is not None and attempt < attempts - 1 else None
            if not creds:
                log_step(ws, step="push", approved=True, ok=False, error=out.error)
                return out
            user, token = creds
            out.error = None
        except (ApprovalRequired, PushError, DockerError, ValueError) as e:
            out.error = str(e)
            on_detail("[ECHEC ] " + out.error)
            log_step(ws, step="push", approved=True, ok=False, error=out.error)
            return out
    user = token = None                               # plus aucune reference au token apres le push
    out.pushed, out.digest = True, r["digest"]
    on_detail(f"[OK    ] pousse : {out.repository}  tags {', '.join(out.tags)}  digest {out.digest}")
    log_step(ws, step="push", approved=True, ok=True, repository=out.repository, tags=out.tags, digest=out.digest)
    return out
