"""Agent 1 — Validator. Fonction pure au sens de l'orchestrateur : (projet, [rapport parent]) -> ValidatorReport scellé.

Python collecte les preuves et statue les contrôles mécaniques. Le LLM n'interprète que les cas ambigus.
Le moteur de politique (package `policy`) calcule le statut d'admission.
Les INVARIANTS ci-dessous (preuves, cohérence, exécution obligatoire, recoupement des faits) sont des paramètres du Validator,
pas des règles de projet.
"""
from __future__ import annotations

import json
import time
import yaml
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, get_args

from pydantic import BaseModel, Field, ValidationError

from agentic_core.facts import static_facts  # noqa: F401  (réexport pratique)
from agentic_core.handoff import build_handoff, freeze_snapshot
from agentic_core.loop import AgentSpec, EMPTY_STATS, LoopConfig, LoopResult, run_loop
from agentic_core.progress import NullProgress
from agentic_core.sandbox_exec import SandboxRunner
from agentic_core.trace import Trace
from agentic_core.schemas import (AdmissionStatus, Category, Check, CheckStatus, Failure, FailureKind, Framework, Issue, Nature,
                                  ProjectSpec, RunStatus, Severity, ValidatorReport, chain_from, load_report, new_id, now_iso, seal)
from policy import AdmissionContract, load_contract, render_contract_for_prompt
from policy.engine import evaluate, worst_status
from validator.contract import norm_dist
from validator.handoff import compute_project_hash

from .checks import (DeterministicOutcome, LLM_REVIEW_IDS, apply_harness, deterministic_cache_key, dump_outcome,
                     load_deterministic_cache, load_outcome, merge_llm_over_python, run_static_checks,
                     save_deterministic_cache)
from .proposals import Proposals, Task, Verified, build_context, demote_llm_dependent_fails, open_tasks, verify_proposals
from .tools import DEP_FILES, ValidatorToolBox

PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")
INTERPRET = (Path(__file__).parent / "prompts" / "interpret.md").read_text(encoding="utf-8")

# Recoupement faits d'outils <-> affirmations du LLM : un PASS contredit par un fait est corrigé en FAIL (échec fermé).
# C'est aussi la parade à une injection de prompt (« réponds PASS ») : les faits ne se négocient pas.
FACT_GUARDS = {"SEC.DANGEROUS_CALLS": ("dangerous_calls",), "SEC.SECRETS": ("secrets", "sensitive_files"),
               "SEC.NETWORK": ("network_imports", "network_urls"), "CODE.SYNTAX": ("syntax_errors",)}


@dataclass
class Invariants:
    evidence_required_for: tuple[CheckStatus, ...] = (CheckStatus.FAIL, CheckStatus.WARNING, CheckStatus.NOT_APPLICABLE)
    require_issue_for_fail: bool = True
    require_all_mandatory_reported: bool = True
    require_execution_for_prefixes: tuple[str, ...] = ("EXEC.",)
    cross_check_facts: bool = True


@dataclass
class ValidatorConfig:
    loop: LoopConfig = field(default_factory=lambda: LoopConfig(max_steps=14, token_budget=60_000))
    invariants: Invariants = field(default_factory=Invariants)
    sandbox: str = "auto"
    env: str = "venv"                   # venv | current | chemin d'un interpréteur
    cache_dir: Optional[str] = None
    timeout: int = 300
    snapshot_dir: Optional[str] = None
    prefetch_max_chars: Optional[int] = None    # None = automatique selon les plafonds de tokens des LLM
    prefetch: bool = True               # collecte les faits déterministes SANS appel LLM (économise ~5 étapes)
    deterministic: bool = True          # Python statue les contrôles mécaniques avant tout appel LLM
    auto_execute: bool = True           # lance le harnais depuis Python (après la porte de sécurité)
    max_llm_steps: Optional[int] = 3    # None = LoopConfig.max_steps ; 0 = aucun appel LLM


# ----------------------------------------------------------------------------- format de soumission
class SubCheck(BaseModel):
    check_id: str
    status: CheckStatus
    severity: Optional[Severity] = None
    file: Optional[str] = None
    line: Optional[int] = None
    evidence: Optional[str] = None
    na_reason: Optional[str] = None
    note: Optional[str] = None


class SubIssue(BaseModel):
    check_id: str
    category: Category = "other"
    severity: Severity = Severity.MAJOR
    file: Optional[str] = None
    line: Optional[int] = None
    description: str
    suggested_fix: Optional[str] = None
    nature: Nature = Nature.MECHANICAL
    evidence: Optional[str] = None


class Submission(BaseModel):
    project_type: str = "unknown"
    frameworks: list[Framework] = Field(default_factory=list)
    checks: list[SubCheck] = Field(default_factory=list)
    issues: list[SubIssue] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    project_spec: Optional[ProjectSpec] = None
    proposals: Optional[Proposals] = None      # HYPOTHÈSES (spec, entrypoints, doutes) : Python les vérifie, elles ne sont jamais un verdict
    summary: Optional[str] = None


def _prefetch_cap(client) -> int:
    """Taille max (caractères) des faits préchargés : réduite si UN des LLM de la chaîne a un plafond de tokens/minute (Groq gratuit),
    sinon la requête dépasserait ce plafond et serait refusée avant l'appel (too_large)."""
    stack, tight = [client], False
    while stack:
        c = stack.pop()
        nested = list(getattr(c, "clients", [])) + ([c.inner] if hasattr(c, "inner") else [])
        if nested:
            stack += nested
        elif getattr(c, "tpm_limit", None):
            tight = True
    return 5200 if (tight and getattr(client, "tpm_limit", None)) else 16000     # lean seulement si le LLM qui répond EN PREMIER a un plafond


def prefetch_facts(tb: ValidatorToolBox, max_chars: int = 16000) -> str:
    """Exécute, sans LLM, les outils déterministes (list_files, analyze_code, dépendances, manifeste, profil des données).
    Passe par tb.call : les sorties comptent comme PREUVES (is_grounded) exactement comme si le LLM les avait demandées."""
    parts = []

    def run(name, args=None):
        out = tb.call(name, args or {})
        parts.append(f"### {name} {json.dumps(args, ensure_ascii=False) if args else ''}\n{out}".replace(" \n", "\n", 1))
        return out

    run("list_files")
    run("analyze_code")
    if any("/" not in f and (f.startswith("requirements") and f.endswith(".txt") or f in DEP_FILES) for f in tb.known_files):
        run("read_dependency_config")      # seulement s'il existe un fichier de dépendances (sinon, c'est au LLM de constater l'absence)
    manifest = next((f for f in ("mlops.yaml", "mlops.yml") if f in tb.known_files), None)
    if manifest:
        run("read_file", {"path": manifest})
        try:
            data = (yaml.safe_load((tb.root / manifest).read_text(encoding="utf-8")) or {}).get("data") or {}
            if isinstance(data.get("path"), str) and data["path"] in tb.known_files:
                run("inspect_data", {"path": data["path"], **({"target": data["target"]} if isinstance(data.get("target"), str) else {})})
        except Exception:
            pass
    root_files = [f for f in tb.known_files if "/" not in f]
    for f in sorted(root_files):                      # dépendances brutes : sans elles, la preuve « scikit-learn==1.8.0 » est introuvable
        if (f.startswith("requirements") and f.endswith(".txt")) or f in DEP_FILES:
            run("read_file", {"path": f})
    for f in sorted((f for f in root_files if f.endswith(".py")), key=lambda n: (n != "train.py", n != "main.py", n)):
        size = int((tb.root / f).stat().st_size * 1.15) + 120          # + en-tête et balises du rendu
        if sum(len(p) for p in parts) + size <= max_chars:              # le code n'est préchargé que s'il reste de la place
            run("read_file", {"path": f})
    return "\n\n".join(parts)


def _format_validation_error(err: dict) -> str:
    loc = ".".join(str(x) for x in err.get("loc") or ())
    msg = err.get("msg") or "invalide"
    if err.get("loc") and err["loc"][-1] == "category":
        allowed = ", ".join(get_args(Category))
        msg = f"{msg} (reçu : {err.get('input')!r}) ; valeurs autorisées pour category : {allowed}"
    return f"format invalide : ({loc}) {msg}"[:400]


def _make_checker(tb: ValidatorToolBox, contract: AdmissionContract, inv: Invariants, *, decided_ids: set[str] | None = None):
    decided_ids = decided_ids or set()

    def check(args: dict):
        try:
            sub = Submission.model_validate(args)
        except ValidationError as e:
            return None, [_format_validation_error(err) for err in e.errors()[:6]]
        problems = []
        ids = {c.check_id for c in sub.checks} | decided_ids
        for c in sub.checks:
            spec = contract.check(c.check_id)
            if spec is None:
                problems.append(f"{c.check_id} : contrôle absent du catalogue du contrat")
                continue
            if c.status in inv.evidence_required_for:
                if not c.evidence:
                    problems.append(f"{c.check_id} {c.status.value} sans evidence")
                elif not tb.is_grounded(c.evidence):
                    problems.append(f"{c.check_id} : evidence introuvable dans les sorties d'outils (copie un extrait exact)")
            elif c.evidence and not tb.is_grounded(c.evidence):
                problems.append(f"{c.check_id} : evidence introuvable dans les sorties d'outils")
            if c.status == CheckStatus.NOT_APPLICABLE and not any(n.id == c.na_reason for n in spec.not_applicable_when):
                problems.append(f"{c.check_id} NOT_APPLICABLE refusé : na_reason doit être l'une de {[n.id for n in spec.not_applicable_when] or 'aucune (non autorisé)'}")
            if c.status == CheckStatus.PASS and any(c.check_id.startswith(p) for p in inv.require_execution_for_prefixes) and not tb.executed_ok:
                problems.append(f"{c.check_id} PASS impossible : le projet n'a pas été exécuté avec succès dans le sandbox (run_in_sandbox) → NOT_CHECKED")
        if inv.require_all_mandatory_reported:
            missing = [i for i in contract.mandatory_ids if i not in ids]
            if missing:
                problems.append(f"contrôles obligatoires non rapportés : {missing} (rapporte-les, au besoin NOT_CHECKED)")
        issue_ids = {i.check_id for i in sub.issues}
        if inv.require_issue_for_fail:
            for c in sub.checks:
                if c.status == CheckStatus.FAIL and c.check_id not in issue_ids:
                    problems.append(f"{c.check_id} FAIL sans problème associé dans `issues`")
        status_of = {c.check_id: c.status for c in sub.checks}
        for i in sub.issues:
            if i.check_id not in status_of:
                problems.append(f"problème sur {i.check_id} : contrôle non rapporté")
            elif status_of[i.check_id] == CheckStatus.PASS:
                problems.append(f"incohérence : {i.check_id} est PASS mais a un problème")
            if not i.evidence:
                problems.append(f"problème sur {i.check_id} sans evidence")
            elif not tb.is_grounded(i.evidence):
                problems.append(f"problème sur {i.check_id} : evidence introuvable dans les sorties d'outils")
        for f in sub.frameworks:
            if f.evidence and not tb.is_grounded(f.evidence):
                # preuve réelle (ligne de dépendance ou d'import) trouvée par l'infrastructure : remplace la citation approximative du LLM
                alt = next((c for c in (f'"name": "{f.name}"', f"{f.name}==", f"import {f.name}", f"from {f.name} ") if tb.is_grounded(c)), None)
                if alt:
                    f.evidence = alt
                else:
                    problems.append(f"framework {f.name} : evidence introuvable (recopie une ligne EXACTE déjà présente dans les sorties, ou omets `evidence`)")
        return sub, problems[:12]
    return check


def _frameworks_from_facts(facts: dict, contract: AdmissionContract) -> list[Framework]:
    third = set(facts.get("third_party_imports") or {})
    in_scope = {norm_dist(x) for x in contract.frameworks_in_scope}
    return [Framework(name=n) for n in sorted(third & in_scope)]


def _llm_needed(cfg: ValidatorConfig, outcome: DeterministicOutcome) -> bool:
    """Le LLM n'est appelé que s'il reste, APRÈS Python, un contrôle obligatoire qu'il peut aider à éclairer : spec non déductible,
    entrypoints ambigus, estimateur inconnu du contrat, sonde de métrique non concluante, ML.METRICS_GENUINE ouvert (voir proposals.open_tasks)."""
    if cfg.max_llm_steps == 0:
        return False
    if not cfg.deterministic:
        return True
    return bool(open_tasks(outcome))


def _python_brief(outcome: DeterministicOutcome) -> str:
    rows = [{"id": c.check_id, "status": c.status.value, "source": c.source, "origin": c.origin,
             "note": (c.note or "")[:160]} for c in outcome.checks]
    open_ids = [c.check_id for c in outcome.checks if c.status == CheckStatus.NOT_CHECKED]
    return json.dumps({"python_checks": rows, "open": open_ids, "security_blocked": outcome.security_blocked,
                       "executed_ok": outcome.executed_ok, "limits": outcome.limits}, ensure_ascii=False)


def _checks_from_submission(sub: Submission, tb: ValidatorToolBox, contract: AdmissionContract, cfg: ValidatorConfig) -> tuple[list[Check], list[Issue]]:
    checks: list[Check] = []
    issues: list[Issue] = []
    facts = tb.facts() if any(c["tool"] == "analyze_code" for c in tb.calls) or tb._facts is not None else {}
    for c in sub.checks:
        d = c.model_dump()
        ev = d.get("evidence")
        d["grounded"] = tb.is_grounded(ev) if ev else None
        d["source"] = "llm"
        note = d.get("note")
        if ev and d["grounded"] is False:
            d["evidence"], note = None, ((note or "") + " [preuve non vérifiée écartée]").strip()
            if c.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE):
                d["status"], note = CheckStatus.NOT_CHECKED, (note + " → NOT_CHECKED (échec fermé)")
        if d["status"] == CheckStatus.PASS and any(c.check_id.startswith(p) for p in cfg.invariants.require_execution_for_prefixes) and not tb.executed_ok:
            d["status"], note = CheckStatus.NOT_CHECKED, "PASS sans exécution réussie dans le sandbox → NOT_CHECKED"
        if cfg.invariants.cross_check_facts and d["status"] == CheckStatus.PASS and any(facts.get(k) for k in FACT_GUARDS.get(c.check_id, ())):
            key = next(k for k in FACT_GUARDS[c.check_id] if facts.get(k))
            d["status"], d["severity"] = CheckStatus.FAIL, Severity.CRITICAL if c.check_id.startswith("SEC.") else Severity.MAJOR
            d["evidence"], note = str(facts[key][0])[:200], f"PASS contredit par analyze_code.{key} → FAIL (recoupement des faits)"
            spec = contract.check(c.check_id)
            issues.append(Issue(issue_id="", check_id=c.check_id, category=spec.category if spec else "other",
                                severity=d["severity"], description=f"Le LLM a déclaré PASS mais l'analyse statique montre {key}.", evidence=d["evidence"]))
        d["note"] = note
        checks.append(Check(**d))
    for i in sub.issues:
        d = i.model_dump()
        if d.get("evidence") and not tb.is_grounded(d["evidence"]):
            d["evidence"] = "[non vérifiée] " + d["evidence"]
        issues.append(Issue(issue_id="", **d))
    if tb.executed_ok is False and any(c.check_id.startswith("EXEC.") and c.status == CheckStatus.PASS for c in sub.checks):
        for c in checks:
            if c.check_id.startswith("EXEC.") and c.status == CheckStatus.PASS:
                c.status, c.note = CheckStatus.FAIL, "PASS contredit par le harnais (executed_ok=false)"
                issues.append(Issue(issue_id="", check_id=c.check_id, category="execution", severity=Severity.MAJOR,
                                    description="L'exécution dans le sandbox a échoué.", evidence=(tb.harness_result or {}).get("_summary")))
    return checks, issues


def _merge_issues(python_issues: list[Issue], llm_issues: list[Issue]) -> list[Issue]:
    by = {i.check_id: i for i in python_issues}
    for i in llm_issues:
        cur = by.get(i.check_id)
        if cur is None:
            by[i.check_id] = i
        elif not cur.suggested_fix and i.suggested_fix:
            cur.suggested_fix = i.suggested_fix
    return list(by.values())


def _user_message_deterministic(root: Path, contract: AdmissionContract, outcome: DeterministicOutcome, tb: ValidatorToolBox,
                                tasks: list[Task] | None = None, max_chars: int = 9000) -> str:
    tasks = open_tasks(outcome) if tasks is None else tasks
    context = build_context(tb, contract, outcome, tasks, max_chars)
    brief = _python_brief(outcome)
    tb.observe(brief)
    return (
        f"Projet à valider : « {root.name} » (profil {contract.id}).\n\n"
        "Python a DÉJÀ statué les contrôles mécaniques. Un FAIL python/harnais est définitif. Tu NE PEUX PAS faire passer un contrôle "
        "obligatoire : tes PASS et FAIL sur ces contrôles sont ignorés (seul NOT_CHECKED + explication est retenu). Python est le seul à "
        "exécuter le projet.\n\n"
        f"{context}\n\n### Détail des contrôles Python\n{brief}\n\n"
        "Ta mission : répondre aux « Questions ouvertes » par des HYPOTHÈSES dans `proposals` (spec, entrypoints, doubts). Python les vérifie "
        "(fichier présent, colonne présente et référencée par le code, fonction existante, exécution réelle) et rejette ce qu'il ne confirme pas. "
        "Dépose submit_report avec `proposals` et, au besoin, les contrôles ouverts en NOT_CHECKED avec une note."
    )


def _check_proposals(tb: ValidatorToolBox, contract: AdmissionContract, cfg: ValidatorConfig, py_out: DeterministicOutcome,
                     proposals: Proposals, log: dict) -> DeterministicOutcome:
    """Vérifie les hypothèses du LLM. Adoption seulement si Python les confirme par observation PUIS par une exécution réelle réussie
    (essai à blanc : si l'exécution échoue, on revient à l'état d'avant, sans rien reprocher au projet)."""
    ver: Verified = verify_proposals(tb, py_out, proposals)
    log["rejected"] += ver.rejected
    log["suggestions"] += ver.suggestions
    if ver.empty():
        return py_out
    saved = (tb.hints, tb._eps, tb.executed_ok, tb.harness_result, tb.sandbox_info, dict(tb._facts or {}))
    tb.hints = {"spec": ver.spec, "entrypoints": ver.entrypoints, "metric_hint": ver.metric_hint, "provenance": ver.provenance}
    tb._eps = None
    trial = None
    try:
        trial = run_static_checks(tb, contract)
        trial = apply_harness(tb, contract, trial, auto_execute=cfg.auto_execute)
    except Exception as e:                                   # jamais d'exception : un essai raté ne devient pas un verdict
        log["rejected"].append(f"essai d'exécution impossible ({type(e).__name__})")
        trial = None
    if trial is None or trial.executed_ok is not True or trial.security_blocked:
        (tb.hints, tb._eps, tb.executed_ok, tb.harness_result, tb.sandbox_info, facts) = saved
        tb._facts = facts or tb._facts
        if trial is not None:
            log["rejected"].append("essai d'exécution avec les valeurs proposées non concluant (le projet ne s'exécute pas avec elles) : propositions écartées")
        return py_out
    demote_llm_dependent_fails(trial)
    if tb._facts is not None:
        tb._facts["entrypoints"] = tb.entrypoint_resolution().summary()
    for k, v in ver.provenance.items():
        log["accepted"].append(f"{k} — {v} ; exécution réelle réussie")
    if ver.metric_hint and (trial.spec is None or trial.spec.primary_metric != ver.metric_hint):
        log["rejected"].append(f"primary_metric « {ver.metric_hint} » : absente du dict renvoyé par train()")
    return trial


def _attach_doubts(checks: list[Check], proposals: Proposals) -> None:
    """Les doutes du LLM éclairent le relecteur : ils s'ajoutent en NOTE d'un contrôle ouvert ; seul un soupçon sur le PASS d'une
    interprétation (ML.METRICS_GENUINE) l'ouvre en NEEDS_REVIEW. Ils ne créent ni PASS ni FAIL."""
    by = {c.check_id: c for c in checks}
    for d in proposals.doubts:
        c = by.get(d.check_id)
        if c is None:
            continue
        txt = f"LLM : {d.explanation.strip()[:240]}"
        if c.status in (CheckStatus.NOT_CHECKED, CheckStatus.WARNING):
            if txt not in (c.note or ""):
                c.note = ((c.note or "") + " · " + txt).strip(" ·")[:600]
        elif d.suspect and c.status == CheckStatus.PASS and c.check_id in LLM_REVIEW_IDS:
            c.status, c.origin = CheckStatus.NOT_CHECKED, "validator"
            c.note = ((c.note or "") + " · soupçon du LLM, non confirmé par Python : " + txt).strip(" ·")[:600]


# ----------------------------------------------------------------------------- construction du rapport
def run_validator(project, client, contract: str | AdmissionContract = "sklearn-tabular", cfg: ValidatorConfig | None = None, *,
                  parent=None, progress=None, runner: SandboxRunner | None = None) -> ValidatorReport:
    """`parent` : rapport précédent (verdict du Judge lors d'une revalidation) -> lignage complet, scellé."""
    cfg = cfg or ValidatorConfig()
    progress = progress or NullProgress()
    contract = contract if isinstance(contract, AdmissionContract) else load_contract(contract)
    root = Path(project).resolve()
    t0 = time.time()
    parent_report = load_report(parent) if parent is not None else None
    base = dict(report_id=new_id(), created_at=now_iso(), project=root.name, policy_version=f"{contract.id}@{contract.version}",
                contract_id=contract.id, lineage=chain_from(parent_report))
    if not root.is_dir():
        pol, _ = evaluate(contract, [], [], executed_ok=False)
        return seal(ValidatorReport(**base, status=AdmissionStatus.NEEDS_REVIEW, next_action="HUMAN_REVIEW", policy=pol,
                                    run_status=RunStatus.FAILED_CLOSED,
                                    failure=Failure(kind=FailureKind.INPUT_INVALID, message=f"{root} n'est pas un dossier"),
                                    limits=["projet introuvable"], duration_s=round(time.time() - t0, 2)))
    trace = Trace(progress, "validator")
    phash, nfiles = compute_project_hash(root)
    tb = ValidatorToolBox(root, contract, runner or SandboxRunner(root, sandbox=cfg.sandbox, env=cfg.env, cache_dir=cfg.cache_dir, timeout=cfg.timeout,
                                                                memory_mb=contract.sandbox_memory_mb or 4096))
    py_out = DeterministicOutcome()
    cache_path = None
    cached_deterministic = False
    if cfg.cache_dir and cfg.deterministic:
        cache_path = Path(cfg.cache_dir) / "deterministic" / f"{deterministic_cache_key(phash, contract, env=cfg.env, sandbox=cfg.sandbox, auto_execute=cfg.auto_execute)}.json"
        hit = load_deterministic_cache(cache_path, project_hash=phash, contract=contract)
        if hit:
            py_out = load_outcome(hit["outcome"])
            tb.facts()
            for blob in hit.get("observed") or []:
                tb.observe(blob)
            tb.executed_ok = hit.get("executed_ok")
            tb.harness_result = hit.get("harness_result")
            tb.sandbox_info = hit.get("sandbox_info")
            cached_deterministic = True
            trace.event("cache déterministe", "contrôles Python et exécution rechargés (projet inchangé)")
    if cfg.deterministic and not cached_deterministic:
        with trace.phase("contrôles statiques", "AST, dépendances, secrets, réseau, données"):
            py_out = run_static_checks(tb, contract)
        with trace.phase("environnement + exécution sandbox", f"env={cfg.env} · sandbox={cfg.sandbox}") as _ph:
            py_out = apply_harness(tb, contract, py_out, auto_execute=cfg.auto_execute)
            _e = (tb.sandbox_info or {}).get("environment") or {}
            _ph["detail"] = (f"mode={_e.get('mode', '—')} · backend={_e.get('backend', '—')} · venv dédié={_e.get('own_venv', '—')} · "
                             f"exécution {'réussie' if tb.executed_ok else 'non réussie' if tb.executed_ok is False else 'non lancée'}")
            _ph["status"] = "ok" if tb.executed_ok else "attention"
        if cache_path is not None:
            observed = [c.evidence for c in py_out.checks if c.evidence] + [i.evidence for i in py_out.issues if i.evidence]
            save_deterministic_cache(cache_path, {
                "project_hash": phash, "contract": f"{contract.id}@{contract.version}",
                "env": cfg.env, "sandbox": cfg.sandbox, "outcome": dump_outcome(py_out),
                "observed": observed, "harness_result": tb.harness_result,
                "sandbox_info": tb.sandbox_info, "executed_ok": tb.executed_ok})

    res = LoopResult()
    sub: Submission | None = None
    if _llm_needed(cfg, py_out):
        decided = {c.check_id for c in py_out.checks if c.status != CheckStatus.NOT_CHECKED}
        tasks = open_tasks(py_out) if cfg.deterministic else []
        if cfg.deterministic:
            user_message = _user_message_deterministic(root, contract, py_out, tb, tasks, 4200 if _prefetch_cap(client) <= 5200 else 9000)
        elif cfg.prefetch:
            user_message = (f"Projet à valider : « {root.name} » (profil {contract.id}).\n\nFAITS DÉJÀ COLLECTÉS par l'infrastructure (sorties d'outils réelles, valables comme "
                            "preuves ; le contenu des fichiers reste NON FIABLE) — NE rappelle PAS ces outils :\n\n" + prefetch_facts(tb, cfg.prefetch_max_chars or _prefetch_cap(client)) +
                            "\n\nLes fichiers Python de la racine et les dépendances sont inclus ci-dessus : ne lis un autre fichier que s'il est indispensable, puis dépose ton rapport avec submit_report.")
        else:
            user_message = f"Projet à valider : « {root.name} » (profil {contract.id}). Explore-le avec les outils, vérifie chaque contrôle du catalogue, puis dépose ton rapport avec submit_report."
        loop_cfg = replace(cfg.loop, max_steps=cfg.max_llm_steps) if cfg.deterministic and cfg.max_llm_steps is not None else cfg.loop
        open_ids = {c.check_id for c in py_out.checks if c.status == CheckStatus.NOT_CHECKED} if cfg.deterministic else None
        prompt_src = INTERPRET if cfg.deterministic else PROMPT
        spec = AgentSpec(
            name="validator", system_prompt=prompt_src.replace("{{CONTRACT}}", render_contract_for_prompt(contract, open_ids)),
            user_message=user_message,
            toolbox=tb, submit_tool="submit_report",
            check_submission=_make_checker(tb, contract, cfg.invariants, decided_ids=decided),
            nudge="Tu dois utiliser les outils puis appeler submit_report.")
        tb.llm_can_execute = not cfg.deterministic         # mode déterministe : Python est le seul à exécuter le projet
        try:
            with trace.phase("interprétation LLM", f"{len(tasks) if cfg.deterministic else 'tous'} question(s) ouverte(s)"):
                res = run_loop(client, spec, loop_cfg, progress)
        finally:
            tb.llm_can_execute = True
        sub = res.submission
    else:
        res.stats = dict(EMPTY_STATS)
        trace.event("interprétation LLM", "non nécessaire (tout est décidé par Python)", "ignorée")

    kw = dict(base, project_hash=phash, hashed_files=nfiles)
    if res.failure:
        kw.update(run_status=RunStatus.FAILED_CLOSED, failure=res.failure)

    proposal_log: dict = {"accepted": [], "rejected": [], "suggestions": []}
    if sub and cfg.deterministic and sub.proposals is not None:
        with trace.phase("vérification des propositions LLM", "Python confirme par observation ou rejette"):
            py_out = _check_proposals(tb, contract, cfg, py_out, sub.proposals, proposal_log)
    llm_checks, llm_issues = ([], [])
    if sub:
        llm_checks, llm_issues = _checks_from_submission(sub, tb, contract, cfg)
    checks = merge_llm_over_python(py_out.checks, llm_checks, frozenset(contract.mandatory_ids)) if cfg.deterministic else llm_checks
    if sub and cfg.deterministic and sub.proposals is not None:
        _attach_doubts(checks, sub.proposals)
    issues = _merge_issues(py_out.issues, llm_issues) if cfg.deterministic else llm_issues
    limits = list(py_out.limits)
    limits += [f"proposition du LLM acceptée après vérification : {x}" for x in proposal_log["accepted"] if x not in limits]
    limits += [f"proposition du LLM rejetée : {x}" for x in proposal_log["rejected"] if x not in limits]
    limits += [f"suggestion du LLM (non tranchée par Python) : {x}" for x in proposal_log["suggestions"] if x not in limits]
    if sub:
        limits += [x for x in sub.limits if x not in limits]
    limits += [x for x in tb.limits_seen if x not in limits]
    if res.failure:
        limits.append(f"analyse LLM interrompue : {res.failure.kind.value}")
    for n, i in enumerate(issues, 1):
        i.issue_id = f"I{n}"

    executed_ok = tb.executed_ok if tb.executed_ok is not None else py_out.executed_ok
    with trace.phase("moteur de politique", f"profil {contract.id}"):
        policy, checks = evaluate(contract, checks, issues, executed_ok=executed_ok)
    status = policy.status
    if kw.get("failure"):                              # échec fermé : jamais mieux que NEEDS_REVIEW
        status = worst_status(status, AdmissionStatus.NEEDS_REVIEW)
        policy = policy.model_copy(update={"status": status, "can_proceed": False,
                                           "reasons": [f"analyse interrompue : {kw['failure'].kind.value}", *policy.reasons]})
    next_action = {AdmissionStatus.ACCEPTED: "PROCEED", AdmissionStatus.REPAIR_REQUIRED: "REPAIR",
                   AdmissionStatus.NEEDS_REVIEW: "HUMAN_REVIEW", AdmissionStatus.REJECTED: "STOP"}[status]
    frameworks = (sub.frameworks if sub and sub.frameworks else _frameworks_from_facts(tb._facts or {}, contract))
    report = ValidatorReport(
        **kw, project_type=(sub.project_type if sub else ("tabular_ml" if py_out.spec else "unknown")),
        frameworks=frameworks, checks=checks, issues=issues,
        limits=limits, project_spec=(py_out.spec if cfg.deterministic else (sub.project_spec if sub and sub.project_spec else py_out.spec)),
        status=status, can_proceed=status == AdmissionStatus.ACCEPTED,
        policy=policy, next_action=next_action, summary=(sub.summary if sub else None),
        observed_metrics=(tb.harness_result or {}).get("metrics"), sandbox=tb.sandbox_info,
        environment=(tb.sandbox_info or {}).get("environment"), steps=trace.export(tb),
        llm={"model": getattr(client, "model", "?"), **res.stats, "tool_calls": len(tb.calls),
             "deterministic": cfg.deterministic, "llm_invoked": bool(res.stats.get("llm_calls")), "proposals": proposal_log,
             "deterministic_cache": cached_deterministic},
        duration_s=round(time.time() - t0, 2))
    if report.can_proceed and cfg.snapshot_dir:
        report.snapshot_path = freeze_snapshot(root, cfg.snapshot_dir, phash)
    report.handoff = build_handoff(report, root)
    return seal(report)
