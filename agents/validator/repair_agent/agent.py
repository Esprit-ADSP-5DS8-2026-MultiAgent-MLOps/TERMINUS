"""Agent 2 — Repair. Fonction : (projet, rapport du Validator) -> RepairReport scellé. Sans état caché : rappelable, reprenable.

L'infrastructure (pas le LLM) fait autorité sur : ce qui a été modifié (historique des patchs), les tests (relancés après la boucle),
l'intégrité de l'original (empreinte avant/après) et la comparaison des métriques avant/après.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from agentic_core.loop import AgentSpec, LoopConfig, run_loop
from agentic_core.progress import NullProgress
from agentic_core.sandbox_exec import SandboxRunner
from agentic_core.schemas import (Failure, FailureKind, Modification, RepairReport, RunStatus, ScopeFlag, UnresolvedIssue,
                                  ValidatorReport, AdmissionStatus, chain_from, load_report, new_id, now_iso, seal, verify_seal)
from policy import load_contract, load_repair_policy
from validator.handoff import compute_project_hash

from .patches import PatchManager
from .runner import compare_metrics, run_checks
from .scope import ScopeGuard
from .tools import RepairToolBox
from .workspace import Workspace, WorkspaceError

PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")


@dataclass
class RepairConfig:
    loop: LoopConfig = field(default_factory=lambda: LoopConfig(max_steps=16, token_budget=60_000, max_tool_calls_per_step=4))
    sandbox: str = "auto"
    env: str = "venv"
    cache_dir: Optional[str] = None
    timeout: int = 300
    run_exec_checks: bool = True          # import + tests unitaires du projet dans le sandbox
    compare_metrics: bool = True


class FixResult(BaseModel):
    issue_id: str
    outcome: Literal["fixed", "not_fixed", "needs_human"]
    patch_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


class FixSubmission(BaseModel):
    results: list[FixResult]
    summary: str = ""


def _checker(issue_ids: set[str], tb: RepairToolBox):
    def check(args):
        try:
            sub = FixSubmission.model_validate(args)
        except ValidationError as e:
            return None, [f"format invalide : {err['loc']} {err['msg']}"[:160] for err in e.errors()[:5]]
        problems, applied = [], {p.patch_id for p in tb.patches.applied}
        got = {r.issue_id for r in sub.results}
        if issue_ids - got:
            problems.append(f"problèmes sans résultat : {sorted(issue_ids - got)}")
        for r in sub.results:
            if r.issue_id not in issue_ids:
                problems.append(f"{r.issue_id} n'est pas un problème reçu")
            if r.outcome == "fixed" and not set(r.patch_ids) & applied:
                problems.append(f"{r.issue_id} 'fixed' sans patch appliqué (patch_ids parmi {sorted(applied) or 'aucun'})")
            if set(r.patch_ids) - applied:
                problems.append(f"{r.issue_id} : patch(s) inconnus ou annulés {sorted(set(r.patch_ids) - applied)}")
        return sub, problems[:8]
    return check


def _base(project: Path, attempt: int, parent) -> dict:
    return dict(repair_id=new_id(), created_at=now_iso(), project=project.name, attempt=attempt, lineage=chain_from(parent))


def _escalated(kw, *, status="ESCALATED", failure=None, unresolved=(), summary="", vstatus=None, src_hash=None, next_action="ESCALATE", **extra):
    return seal(RepairReport(**kw, status=status, source_project_hash=src_hash, source_validator_status=vstatus, unresolved_issues=list(unresolved),
                             requires_human_approval=any(u.needs_human for u in unresolved), next_action=next_action, summary=summary,
                             run_status=RunStatus.FAILED_CLOSED if failure else RunStatus.OK, failure=failure, **extra))


def run_repair(project, validator_report, output_dir, client, cfg: RepairConfig | None = None, *, attempt: int = 1,
               judge_feedback=None, progress=None, runner: SandboxRunner | None = None) -> RepairReport:
    cfg, progress, policy = cfg or RepairConfig(), progress or NullProgress(), load_repair_policy()
    src = Path(project).resolve()
    try:
        vr = load_report(validator_report)
    except Exception as e:
        return _escalated(_base(src, attempt, None), status="FAILED", failure=Failure(kind=FailureKind.INPUT_INVALID, message=f"rapport illisible : {type(e).__name__}"))
    kw = _base(src, attempt, vr if isinstance(vr, ValidatorReport) else None)
    if not isinstance(vr, ValidatorReport):
        return _escalated(kw, status="FAILED", failure=Failure(kind=FailureKind.INPUT_INVALID, message="un rapport du Validator est attendu"))
    vstatus = vr.status.value
    if not verify_seal(vr):
        return _escalated(kw, status="FAILED", vstatus=vstatus, failure=Failure(kind=FailureKind.INPUT_INVALID, message="rapport du Validator modifié (sceau invalide)"))
    if not src.is_dir():
        return _escalated(kw, status="FAILED", vstatus=vstatus, failure=Failure(kind=FailureKind.INPUT_INVALID, message=f"{src} n'est pas un dossier"))
    src_hash, _ = compute_project_hash(src)
    if vr.project_hash and vr.project_hash != src_hash:
        return _escalated(kw, status="FAILED", vstatus=vstatus, src_hash=src_hash,
                          failure=Failure(kind=FailureKind.INPUT_INVALID, message="le projet a changé depuis la validation (empreinte différente) : revalidez d'abord"))
    if vr.run_status == RunStatus.FAILED_CLOSED:
        un = [UnresolvedIssue(issue_id=i.issue_id, reason="analyse du Validator interrompue : rapport partiel, non fiable pour réparer", needs_human=False) for i in vr.issues]
        return _escalated(kw, vstatus=vstatus, src_hash=src_hash, unresolved=un, summary="le Validator a échoué en cours d'analyse (" +
                          (vr.failure.kind.value if vr.failure else "?") + ") : relancez la validation avant de réparer")
    if vr.status == AdmissionStatus.ACCEPTED:
        return _escalated(kw, status="NOTHING_TO_DO", vstatus=vstatus, src_hash=src_hash, next_action="NONE", summary="projet déjà accepté : rien à réparer")
    if vr.status == AdmissionStatus.REJECTED:
        un = [UnresolvedIssue(issue_id=i.issue_id, reason="projet REJETÉ (hors périmètre ou non sûr) : non réparable automatiquement", needs_human=True) for i in vr.issues]
        return _escalated(kw, vstatus=vstatus, src_hash=src_hash, unresolved=un, summary="rejeté : aucune réparation tentée")

    # ---- triage : seules les natures autorisées sont confiées au LLM ; la sécurité n'est jamais « réparée » par un LLM
    auto, unresolved = [], []
    for i in vr.issues:
        if i.category == "security" and i.check_id != "SEC.COMMITTED_ARTIFACTS":
            unresolved.append(UnresolvedIssue(issue_id=i.issue_id, needs_human=True, reason="problème de sécurité : jamais réparé automatiquement"))
        elif i.nature.value not in policy.auto_repair_natures:
            unresolved.append(UnresolvedIssue(issue_id=i.issue_id, needs_human=True, reason=f"nature « {i.nature.value} » : décision humaine requise (choix de logique ou de méthode)"))
        else:
            auto.append(i)
    if not auto:
        return _escalated(kw, vstatus=vstatus, src_hash=src_hash, unresolved=unresolved,
                          summary="aucun problème réparable automatiquement" + ("" if vr.issues else " (NEEDS_REVIEW sans problème listé : revue humaine)"))

    try:
        ws = Workspace.create(src, output_dir)
    except WorkspaceError as e:
        return _escalated(kw, status="FAILED", vstatus=vstatus, src_hash=src_hash, unresolved=unresolved,
                          failure=Failure(kind=FailureKind.INPUT_INVALID, message=str(e)))
    t0 = time.time()
    patches = PatchManager(ws.root, ws.meta)
    runner = runner or SandboxRunner(ws.root, sandbox=cfg.sandbox, env=cfg.env, cache_dir=cfg.cache_dir, timeout=cfg.timeout)
    tb = RepairToolBox(ws, patches, ScopeGuard(policy), runner if cfg.run_exec_checks else None)

    artifact_issues = [i for i in auto if i.check_id == "SEC.COMMITTED_ARTIFACTS"]
    todo = [i for i in auto if i not in artifact_issues]
    fixed_by_copy = [i for i in artifact_issues if ws.removed_artifacts]           # artefacts binaires non copiés = corrigés
    for i in artifact_issues:
        if i not in fixed_by_copy:
            todo.append(i)

    res, sub = None, None
    if todo:
        feedback = ""
        if judge_feedback is not None:
            fb = load_report(judge_feedback)
            feedback = f"\n\nTENTATIVE PRÉCÉDENTE REFUSÉE PAR LE JUDGE. Préoccupations : {json.dumps(fb.remaining_concerns, ensure_ascii=False)} ; raison : {fb.reason}"
        payload = [{"issue_id": i.issue_id, "check_id": i.check_id, "severity": i.severity.value, "file": i.file, "line": i.line,
                    "description": i.description, "suggested_fix": i.suggested_fix, "evidence": i.evidence} for i in todo]
        spec = AgentSpec(name="repair", system_prompt=PROMPT,
                         user_message=f"Projet « {src.name} » (type : {vr.project_type}). Problèmes à corriger dans la copie de travail :\n{json.dumps(payload, ensure_ascii=False)}{feedback}",
                         toolbox=tb, submit_tool="submit_repair", check_submission=_checker({i.issue_id for i in todo}, tb),
                         nudge="Utilise les outils (list_files, read_file, replace_in_file…) puis appelle submit_repair.")
        res = run_loop(client, spec, cfg.loop, progress)
        sub = res.submission

    failure = res.failure if res else None
    if failure:                                           # échec fermé : on annule tout, rien n'est déclaré corrigé
        patches.rollback_all()
    flags: list[ScopeFlag] = list(tb.flags)
    outcomes = {r.issue_id: r for r in (sub.results if sub and not failure else [])}
    for i in todo:
        r = outcomes.get(i.issue_id)
        if r is None:
            unresolved.append(UnresolvedIssue(issue_id=i.issue_id, reason="non traité" + (f" ({failure.kind.value})" if failure else ""), needs_human=False))
        elif r.outcome != "fixed":
            unresolved.append(UnresolvedIssue(issue_id=i.issue_id, reason=r.explanation or r.outcome, needs_human=r.outcome == "needs_human"))
    mods = patches.modifications()
    for p in ws.removed_artifacts:
        mods.append(Modification(patch_id="artifact", file=p, action="delete", reason="artefact binaire non copié dans la copie de travail (jamais chargé)",
                                 issue_id=(fixed_by_copy[0].issue_id if fixed_by_copy else None)))
    if not ws.source_untouched():
        flags.append(ScopeFlag(file="<projet source>", kind="source_modified", detail="l'empreinte du projet original a changé : INCIDENT"))
        failure = failure or Failure(kind=FailureKind.INTERNAL, message="l'original a été modifié")

    tests = run_checks(ws.root, runner if cfg.run_exec_checks else None)
    vspec = vr.project_spec
    mc = compare_metrics(src, ws.root, runner if (cfg.compare_metrics and cfg.run_exec_checks) else None, vspec, load_contract(vr.contract_id).harness if vr.contract_id else None, policy.metric_tolerance) \
        if mods and not failure else None
    approval = any(not f.blocked and f.kind in ("scientific_change", "hardcoded_metric") for f in flags) or any(u.needs_human for u in unresolved) or \
        bool(mc and mc.behavior_changed)
    changed = bool(mods)
    if failure:
        status = "FAILED"
    elif not changed:
        status = "ESCALATED"
    elif unresolved or any(t.status == "FAIL" for t in tests):
        status = "PARTIAL"
    else:
        status = "REPAIRED"
    (ws.meta / "full.diff").write_text(patches.full_diff(), encoding="utf-8")
    report = RepairReport(
        **kw, status=status, source_project_hash=src_hash, repaired_project_hash=ws.hash() if not failure else None,
        workspace_path=str(ws.root), meta_path=str(ws.meta), source_validator_status=vstatus, modifications=mods, patches=patches.history,
        tests=tests, unresolved_issues=unresolved, scope_flags=flags, requires_human_approval=approval, metrics_comparison=mc,
        requires_judge=changed and not failure, requires_revalidation=changed and not failure,
        next_action="JUDGE" if changed and not failure else "ESCALATE", removed_artifacts=ws.removed_artifacts, diff=patches.full_diff(),
        summary=(sub.summary if sub else "") or ("réparation échouée" if failure else ""),
        run_status=RunStatus.FAILED_CLOSED if failure else RunStatus.OK, failure=failure,
        llm={"model": getattr(client, "model", "?"), **(res.stats if res else {}), "tool_calls": len(tb.calls), "duration_s": round(time.time() - t0, 2)})
    return seal(report)


def rollback_workspace(report: RepairReport) -> bool:
    """Action `ROLLBACK` demandée par le Judge : restaure la copie de travail à l'état de l'original (l'original, lui, n'a jamais bougé)."""
    import shutil
    if not report.workspace_path:
        return False
    shutil.rmtree(report.workspace_path, ignore_errors=True)
    return not Path(report.workspace_path).exists()
