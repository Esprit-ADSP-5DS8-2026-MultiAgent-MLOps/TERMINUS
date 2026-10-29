from __future__ import annotations

import time
from pathlib import Path

from . import check_data, check_dynamic, check_manifest, check_static, check_structure
from .context import Context
from .contract import CONTRACT_VERSION, ContractConfig
from .fs import walk_project
from .models import LevelInfo, Severity as S, ValidationReport, Verdict


def _verdict(findings) -> Verdict:
    sev = {f.severity for f in findings}
    if S.BLOCKER in sev:
        return Verdict.REJECTED
    if S.CONFIRM in sev:
        return Verdict.NEEDS_CONFIRMATION
    if S.WARNING in sev:
        return Verdict.ACCEPTED_WITH_WARNINGS
    return Verdict.ACCEPTED


def validate(project, cfg: ContractConfig | None = None, *, dynamic: bool = True, sandbox: str = "auto",
             determinism: bool = True, bandit: bool = True, timeout: int | None = None, keep_workdir: bool = False,
             env: str = "venv", cache_dir: str | None = None,
             snapshot_dir: str | None = None, confirmed_by: str | None = None, confirm_note: str | None = None) -> ValidationReport:
    t_start = time.time()
    root = Path(project).resolve()
    cfg = cfg or ContractConfig()
    ctx = Context(root=root, cfg=cfg, options={"sandbox": sandbox, "determinism": determinism, "bandit": bandit,
                                              "timeout": timeout or cfg.dynamic_timeout_s, "keep_workdir": keep_workdir, "env": env, "cache_dir": cache_dir})
    levels: list[LevelInfo] = []

    def level(name, fn, skip_reason=None):
        ctx.level = name
        if skip_reason:
            levels.append(LevelInfo(name=name, status="skipped", reason=skip_reason))
            return
        t0 = time.time()
        try:
            fn(ctx)
        except Exception as e:  # un bug du validateur ne doit jamais produire un faux "ACCEPTED"
            ctx.add("ENGINE", "ENGINE.CHECK_CRASHED", S.BLOCKER, f"Le niveau `{name}` a planté : {type(e).__name__}: {e}")
        levels.append(LevelInfo(name=name, status="done", duration_s=round(time.time() - t0, 3)))

    if not root.is_dir():
        ctx.level = "structure"
        ctx.add("C1", "C1.NOT_A_DIRECTORY", S.BLOCKER, f"Dossier introuvable : {root}")
    else:
        ctx.files, ctx.symlinks = walk_project(root)
        level("structure", check_structure.run)
        level("manifest", check_manifest.run)
        level("static", check_static.run)
        level("data", check_data.run)
        if not dynamic:
            reason = "désactivé (--no-dynamic)"
        elif ctx.has(S.BLOCKER):
            reason = "bloquants aux niveaux 1-4 : le code du projet n'est pas exécuté"
        elif ctx.manifest is None:
            reason = "manifest absent : cible inconnue, confirmez d'abord mlops.yaml"
        else:
            reason = None
        level("dynamic", check_dynamic.run, skip_reason=reason)

    order = {S.BLOCKER: 0, S.CONFIRM: 1, S.WARNING: 2, S.INFO: 3}
    findings = sorted(ctx.findings, key=lambda f: order[f.severity])
    summary = {s.value: sum(1 for f in findings if f.severity == s) for s in S}
    report = ValidationReport(
        contract_version=CONTRACT_VERSION, project=root.name, verdict=_verdict(findings), summary=summary, findings=findings,
        levels=levels, facts=ctx.facts, observed_metrics=ctx.observed_metrics, sandbox=ctx.sandbox_info,
        duration_s=round(time.time() - t_start, 2))
    from .handoff import finalize  # import tardif : permet `python -m validator.handoff` sans avertissement
    return finalize(report, ctx, snapshot_dir=snapshot_dir, confirmed_by=confirmed_by, confirm_note=confirm_note)
