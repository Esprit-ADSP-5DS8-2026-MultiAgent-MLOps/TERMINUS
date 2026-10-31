"""Collecte de trajectoires du Repair : projet sain -> défaut injecté -> Validator -> Repair (enregistré) -> revalidation -> (Judge) -> filtres.

    GROQ_API_KEY=... python -m finetune.collect --out data/run1 --bases 10 --fault-seeds 2 --samples 2 --judge --temperature 0.7
    python -m finetune.collect --out data/check --bases 2 --check-faults          # sans LLM : vérifie injection + routage vers le Repair

Écrit `runs.jsonl` : UNE ligne par exécution (retenue OU écartée, avec ses raisons), trajectoire complète incluse. Reprenable (--out identique).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agentic_core.llm_client import LLMError
from finetune.bases import make_base
from finetune.faults import FAULTS, inject
from finetune.filters import FilterConfig, changed_lines, evaluate, invented_paths, repair_filters, revalidation_filters
from finetune.recorder import RecordingClient, rebuild_trajectory


class NoLLM:
    """Client du Validator : le Validator décide en Python (max_llm_steps=0) ; tout appel LLM échouerait fermé."""
    model = "none"

    def chat(self, messages, tools=None):
        raise LLMError("aucun LLM pour le Validator pendant la collecte", "missing_key")


@dataclass
class CollectConfig:
    out_dir: Path
    bases: int = 5
    seed0: int = 0
    faults: list[str] = field(default_factory=lambda: list(FAULTS))
    fault_seeds: int = 1
    samples: int = 1
    env: str = "venv"                       # venv | current | interpréteur : environnement d'exécution du Validator et du Repair
    contract: str = "sklearn-tabular"
    judge: bool = False
    validator_llm: bool = False             # False : le Validator statue en Python seul (gratuit, déterministe)
    filters: FilterConfig = field(default_factory=FilterConfig)
    keep_workdirs: bool = False
    check_faults: bool = False              # n'appelle aucun LLM : vérifie seulement injection + routage


def gold_lines(base: Path, proj: Path, files: tuple[str, ...]) -> int:
    n = 0
    for f in files:
        a = (base / f).read_text(encoding="utf-8").splitlines() if (base / f).is_file() else []
        try:
            b = (proj / f).read_text(encoding="utf-8").splitlines() if (proj / f).is_file() else []
        except UnicodeDecodeError:
            n += 1
            continue
        n += sum(1 for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    return n


def _repair_summary(r, client: RecordingClient, out: Path) -> dict:
    added, removed, files = changed_lines(r.diff or "")
    mc = r.metrics_comparison
    return {"status": r.status, "next_action": r.next_action, "failure": r.failure.kind.value if r.failure else None,
            "tests": [{"name": t.name, "status": t.status} for t in r.tests], "scope_flags": [f.kind for f in r.scope_flags],
            "requires_human_approval": r.requires_human_approval, "behavior_changed": mc.behavior_changed if mc else None,
            "unresolved_ids": [u.issue_id for u in r.unresolved_issues], "files_changed": files, "added": added, "removed": removed,
            "removed_artifacts": list(r.removed_artifacts), "steps": len(client.records), "model": getattr(client, "model", None),
            "invented_paths": invented_paths(r.diff or "", out) if out.is_dir() else [], "summary": r.summary}


def run_sample(*, proj: Path, vr, fault, base_dir: Path, meta: dict, base_checks: set, base_status: str, run_id: str, k: int, cfg: CollectConfig, work: Path,
               repair_client, judge_client_factory, validator_client, vcfg) -> dict:
    from repair_agent import RepairConfig, run_repair
    from validator_agent import run_validator
    out = work / f"fixed_{k}"
    client = RecordingClient(repair_client)
    t0 = time.time()
    r = run_repair(proj, vr, out, client, RepairConfig(env=cfg.env), attempt=1)
    base_ids = [i.issue_id for i in vr.issues if i.check_id in base_checks]
    run = {"run_id": run_id, "base_seed": meta["seed"], "fault_id": fault.id, "sample": k, "base": meta, "baseline_checks": sorted(base_checks),
           "baseline_status": base_status, "baseline_issue_ids": base_ids,
           "issues": [{"issue_id": i.issue_id, "check_id": i.check_id, "nature": i.nature.value, "file": i.file, "severity": i.severity.value} for i in vr.issues],
           "gold": {"files": list(fault.gold_files), "changed_lines": gold_lines(base_dir, proj, fault.gold_files)},
           "repair": _repair_summary(r, client, out), "revalidation": None, "judge": None}
    traj = rebuild_trajectory(client.records)
    if traj:
        traj["tools"] = client.tools
    run["trajectory"] = traj
    reasons = repair_filters(run, cfg.filters)
    if not reasons and r.workspace_path:
        v2 = run_validator(out, validator_client, cfg.contract, vcfg)
        run["revalidation"] = {"status": v2.status.value, "issue_checks": sorted({i.check_id for i in v2.issues})}
        reasons += revalidation_filters(run)
        if not reasons and cfg.judge and judge_client_factory:
            from repair_agent.judge import JudgeConfig, run_judge
            j = run_judge(proj, r, judge_client_factory(), vr, JudgeConfig(require_distinct_model=True))
            run["judge"] = {"verdict": j.verdict, "next_action": j.next_action, "independent_model": j.independent_model,
                            "model": j.judge_model, "failure": j.failure.kind.value if j.failure else None}
    run["rejected_by"] = evaluate(run, cfg.filters)
    run["accepted"] = not run["rejected_by"]
    run["duration_s"] = round(time.time() - t0, 1)
    return run


def collect(cfg: CollectConfig, repair_client_factory: Callable | None, judge_client_factory: Callable | None = None, log=print) -> dict:
    from validator_agent import ValidatorConfig, run_validator
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path, check_path = cfg.out_dir / "runs.jsonl", cfg.out_dir / "faults_check.jsonl"
    done = {json.loads(l)["run_id"] for l in runs_path.read_text(encoding="utf-8").splitlines() if l.strip()} if runs_path.exists() else set()
    vcfg = ValidatorConfig(env=cfg.env, max_llm_steps=None if cfg.validator_llm else 0)
    vclient = NoLLM()
    stats = {"tasks": 0, "skipped": 0, "runs": 0, "accepted": 0, "errors": 0}
    root = Path(tempfile.mkdtemp(prefix="finetune_"))
    try:
        for b in range(cfg.seed0, cfg.seed0 + cfg.bases):
            base_dir = root / f"base_{b}" / "healthy"
            meta = make_base(base_dir, b)
            bv = run_validator(base_dir, vclient, cfg.contract, vcfg)
            base_checks = {i.check_id for i in bv.issues}
            for fid in cfg.faults:
                for fs in range(cfg.fault_seeds):
                    stats["tasks"] += 1
                    proj = root / f"task_{b}_{fid}_{fs}" / meta["domain"]      # le prompt du Repair contient le nom du dossier : jamais l'id du défaut
                    shutil.copytree(base_dir, proj)
                    try:
                        fault = inject(fid, proj, fs, meta)
                        vr = run_validator(proj, vclient, cfg.contract, vcfg)
                    except Exception as e:                       # une tâche cassée ne doit pas arrêter une collecte longue
                        stats["errors"] += 1
                        log(f"[collect] erreur d'injection {fid}: {type(e).__name__}: {e}", file=sys.stderr)
                        continue
                    new = sorted({i.check_id for i in vr.issues if i.check_id not in base_checks})
                    routed = vr.next_action == "REPAIR" and bool(set(new) & set(fault.expected_checks))
                    task = {"base_seed": b, "fault_id": fid, "fault_seed": fs, "validator_status": vr.status.value, "next_action": vr.next_action,
                            "new_checks": new, "expected_checks": list(fault.expected_checks), "routed": routed}
                    if cfg.check_faults:
                        with check_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(task, ensure_ascii=False) + "\n")
                        log(f"[check] base{b} {fid:22s} {vr.status.value:16s} routé={routed} {new}")
                        continue
                    if not routed:
                        stats["skipped"] += 1
                        log(f"[collect] ignoré {fid} base{b}: pas routé vers le Repair ({vr.next_action}, {new})")
                        continue
                    for k in range(cfg.samples):
                        rid = f"{b}-{fid}-{fs}-s{k}"
                        if rid in done:
                            continue
                        try:
                            run = run_sample(proj=proj, vr=vr, fault=fault, base_dir=base_dir, meta=meta, base_checks=base_checks, base_status=bv.status.value, run_id=rid, k=k, cfg=cfg,
                                             work=proj.parent / "work", repair_client=repair_client_factory(), judge_client_factory=judge_client_factory,
                                             validator_client=vclient, vcfg=vcfg)
                        except KeyboardInterrupt:
                            log("[collect] interrompu : relancez la même commande pour reprendre", file=sys.stderr)
                            return stats
                        except Exception as e:
                            stats["errors"] += 1
                            log(f"[collect] erreur {rid}: {type(e).__name__}: {e}", file=sys.stderr)
                            continue
                        with runs_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(run, ensure_ascii=False) + "\n")
                        stats["runs"] += 1
                        stats["accepted"] += int(run["accepted"])
                        log(f"[collect] {rid:28s} {'RETENU ' if run['accepted'] else 'écarté '} {run['repair']['steps']} étapes  {','.join(run['rejected_by'])}")
    finally:
        if not cfg.keep_workdirs:
            shutil.rmtree(root, ignore_errors=True)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True), ap.add_argument("--bases", type=int, default=5), ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--faults", nargs="*", help=f"parmi : {', '.join(FAULTS)} (défaut : tous)")
    ap.add_argument("--fault-seeds", type=int, default=1, help="variantes aléatoires de chaque défaut"), ap.add_argument("--samples", type=int, default=1, help="essais du Repair par tâche")
    ap.add_argument("--env", default="venv"), ap.add_argument("--judge", action="store_true"), ap.add_argument("--validator-llm", action="store_true")
    ap.add_argument("--model", help="modèle du Repair"), ap.add_argument("--temperature", type=float, default=0.7, help="0 = déterministe (échantillons identiques)")
    ap.add_argument("--allow-unjudged", action="store_true"), ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--check-faults", action="store_true"), ap.add_argument("--keep-workdirs", action="store_true")
    a = ap.parse_args(argv)
    unknown = set(a.faults or []) - set(FAULTS)
    if unknown:
        print(f"défauts inconnus : {sorted(unknown)}", file=sys.stderr)
        return 2
    cfg = CollectConfig(out_dir=Path(a.out), bases=a.bases, seed0=a.seed0, faults=a.faults or list(FAULTS), fault_seeds=a.fault_seeds, samples=a.samples, env=a.env,
                        judge=a.judge, validator_llm=a.validator_llm, keep_workdirs=a.keep_workdirs, check_faults=a.check_faults,
                        filters=FilterConfig(require_judge=not a.allow_unjudged, max_steps=a.max_steps))
    if cfg.check_faults:
        print(json.dumps(collect(cfg, None), indent=2))
        return 0
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY manquante (ou utilisez --check-faults).", file=sys.stderr)
        return 2
    if cfg.filters.require_judge and not cfg.judge:
        print("Sans Judge, ajoutez --judge (recommandé) ou --allow-unjudged : un Validator qui accepte ne prouve pas que la réparation est réelle.", file=sys.stderr)
        return 2
    from agentic_core.llm_client import GroqClient, build_client
    repair = (lambda: GroqClient.for_agent("repair", a.model, temperature=a.temperature)) if a.temperature > 0 else (lambda: build_client("repair", a.model))
    stats = collect(cfg, repair, (lambda: build_client("judge")) if cfg.judge else None)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
