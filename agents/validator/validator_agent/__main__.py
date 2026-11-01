"""CLI : python -m validator_agent PROJET [--contract sklearn-tabular] [--json rapport.json] [--parent verdict.json]"""
import argparse
import sys
from pathlib import Path

from agentic_core.llm_client import build_client
from agentic_core.progress import ConsoleProgress, NullProgress
from agentic_core.schemas import AdmissionStatus, RunStatus, schema_json  # noqa: F401
from policy import list_contracts

from .agent import ValidatorConfig, run_validator

EXIT = {AdmissionStatus.ACCEPTED: 0, AdmissionStatus.NEEDS_REVIEW: 10, AdmissionStatus.REPAIR_REQUIRED: 20, AdmissionStatus.REJECTED: 30}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="validator_agent", description="Agent Validator (LLM + moteur de politique)")
    ap.add_argument("project")
    ap.add_argument("--contract", default="sklearn-tabular", help=f"profil ({', '.join(list_contracts())}) ou chemin d'un YAML")
    ap.add_argument("--json", help="écrit le rapport scellé (JSON)")
    ap.add_argument("--parent", help="rapport précédent (verdict du Judge) : revalidation avec lignage")
    ap.add_argument("--model"), ap.add_argument("--env", default="venv", help="venv | current | chemin d'un interpréteur Python")
    ap.add_argument("--sandbox", default="auto", choices=["auto", "docker", "subprocess"])
    ap.add_argument("--snapshot-dir", help="copie figée du projet si ACCEPTED")
    ap.add_argument("--cache-dir", help="cache des venvs"), ap.add_argument("--llm-cache", help="cassettes LLM (enregistrement/rejeu)")
    ap.add_argument("--replay", action="store_true"), ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--no-prefetch", action="store_true", help="(mode --explore) le LLM appelle lui-même tous les outils")
    ap.add_argument("--no-llm", action="store_true", help="aucun appel LLM : contrôles Python/harnais uniquement")
    ap.add_argument("--max-llm-steps", type=int, default=3, help="plafond d'appels LLM (0 = --no-llm)")
    ap.add_argument("--steps", action="store_true", help="affiche le détail des étapes effectuées (phases + appels d'outils)")
    ap.add_argument("--explore", action="store_true", help="ancien parcours : le LLM orchestre les outils (plus de tokens)")
    return ap


def config_from_args(a: argparse.Namespace) -> ValidatorConfig:
    return ValidatorConfig(sandbox=a.sandbox, env=a.env, cache_dir=a.cache_dir, snapshot_dir=a.snapshot_dir,
                           prefetch=not a.no_prefetch, deterministic=not a.explore,
                           max_llm_steps=0 if a.no_llm else a.max_llm_steps)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    client = build_client("validator", a.model, a.llm_cache, a.replay)
    cfg = config_from_args(a)
    rep = run_validator(a.project, client, a.contract, cfg,
                        parent=a.parent, progress=NullProgress() if a.quiet else ConsoleProgress())
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n{rep.status.value}  (can_proceed={rep.can_proceed} · profil {rep.contract_id} · prochaine action : {rep.next_action})")
    reasons = rep.policy.reasons if rep.policy else []
    for r in reasons[:6]:
        print(f"  • {r}")
    if len(reasons) > 6:
        print(f"  • … et {len(reasons) - 6} autre(s) raison(s) (voir le JSON)")
    for i in rep.issues:
        print(f"  [{i.severity.value}] {i.issue_id} {i.check_id} ({i.nature.value}) {i.file or ''} — {i.description}")
    for l in rep.limits:
        print(f"  limite : {l}")
    e = rep.environment
    if e:
        ok = next((c for c in rep.checks if c.check_id == "ENV.SANDBOX_ISOLATION"), None)
        print(f"\nEnvironnement d'exécution : {ok.status.value if ok else '—'} · mode {e.get('mode')} · {e.get('backend')} · python {e.get('python_version', '?')} "
              f"· venv dédié : {'oui' if e.get('own_venv') else 'non'} · dépendances : {e.get('deps_source') or 'aucune'} "
              f"({e.get('pinned_checked', 0)} épinglée(s) vérifiée(s), {len(e.get('mismatches') or []) + len(e.get('missing') or [])} écart(s)) "
              f"· réseau isolé : {'oui' if e.get('network_isolated') else 'non'}")
    if a.steps:
        print("\nÉtapes effectuées :")
        for s in rep.steps:
            print(f"  {s['n']:>2}. t+{s['t']:>6.1f}s {s['duration_s']:>6.1f}s  {'PHASE' if s['kind'] == 'phase' else 'outil':<5}  {s['name']}"
                  + (f" — {s['detail']}" if s['detail'] else "") + ("" if s['status'] == 'ok' else f"  [{s['status']}]"))
    if rep.failure:
        print(f"  ÉCHEC FERMÉ : {rep.failure.kind.value} — {rep.failure.message}")
    return EXIT[rep.status]


if __name__ == "__main__":
    sys.exit(main())
