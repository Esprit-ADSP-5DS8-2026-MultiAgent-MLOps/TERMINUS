from __future__ import annotations

import argparse
import sys

from .contract import CONTRACT_VERSION, ContractConfig, norm_dist
from .engine import validate
from .models import Verdict

EXIT = {Verdict.ACCEPTED: 0, Verdict.ACCEPTED_WITH_WARNINGS: 0, Verdict.NEEDS_CONFIRMATION: 10, Verdict.REJECTED: 20}
ICON = {"BLOCKER": "✖", "CONFIRM": "?", "WARNING": "!", "INFO": "i"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m validator", description=f"Contract Validator v{CONTRACT_VERSION}")
    ap.add_argument("project", help="dossier du projet ML à valider")
    ap.add_argument("--json", metavar="FICHIER", help="écrit le rapport JSON")
    ap.add_argument("--no-dynamic", action="store_true", help="n'exécute jamais le code (niveaux 1-4 seulement)")
    ap.add_argument("--sandbox", choices=["auto", "docker", "subprocess"], default="auto")
    ap.add_argument("--timeout", type=int, default=None, help="secondes (défaut : 300)")
    ap.add_argument("--no-determinism", action="store_true", help="saute le 2e entraînement de contrôle (C9)")
    ap.add_argument("--no-bandit", action="store_true")
    ap.add_argument("--env", choices=["venv", "current"], default="venv",
                    help="venv (défaut) : environnement isolé construit depuis requirements.txt ; current : Python courant")
    ap.add_argument("--cache-dir", default=None, help="cache des venv (défaut : ~/.cache/contract-validator)")
    ap.add_argument("--allow-framework", action="append", default=[], help="ex. xgboost (élargit le contrat)")
    ap.add_argument("--snapshot-dir", default=None, help="copie en lecture seule du projet validé (si le passage de relais est autorisé)")
    ap.add_argument("--confirmed-by", default=None, help="nom de la personne qui a confirmé le manifest (trace, ne change pas le verdict)")
    ap.add_argument("--confirm-note", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    cfg = ContractConfig(allow_frameworks=frozenset(norm_dist(x) for x in a.allow_framework))
    rep = validate(a.project, cfg, snapshot_dir=a.snapshot_dir, confirmed_by=a.confirmed_by, confirm_note=a.confirm_note, dynamic=not a.no_dynamic, sandbox=a.sandbox, determinism=not a.no_determinism,
                   bandit=not a.no_bandit, timeout=a.timeout, env=a.env, cache_dir=a.cache_dir)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(rep.model_dump_json(indent=2))
    if not a.quiet:
        print(f"Contract Validator v{rep.contract_version} — projet : {rep.project}")
        print(f"Verdict : {rep.verdict.value}   ({rep.summary['BLOCKER']} bloquant(s), {rep.summary['CONFIRM']} confirmation(s), "
              f"{rep.summary['WARNING']} avertissement(s))   [{rep.duration_s}s]")
        print("Niveaux : " + " | ".join(f"{l.name} {'✔' if l.status == 'done' else '— ignoré (' + (l.reason or '') + ')'}" for l in rep.levels))
        if rep.handoff:
            why = f" — {rep.handoff.reason}" if rep.handoff.reason else ""
            print(f"Passage de relais : {rep.handoff.status} (peut continuer : {'oui' if rep.handoff.can_proceed else 'non'}){why}")
        print(f"Run : {rep.run_id} | empreinte du projet : {(rep.project_hash or '-')[:16]}… ({rep.hashed_files} fichiers)")
        if rep.snapshot_path:
            print(f"Copie validée : {rep.snapshot_path}")
        if rep.sandbox:
            print(f"Sandbox : {rep.sandbox['backend']} (réseau isolé : {'oui' if rep.sandbox['network_isolated'] else 'non'})")
        if rep.observed_metrics:
            print("Métriques observées : " + ", ".join(f"{k}={v:.4f}" for k, v in rep.observed_metrics.items()))
        for f in rep.findings:
            loc = f" {f.file}{':' + str(f.line) if f.line else ''}" if f.file else ""
            print(f"  [{ICON[f.severity.value]} {f.severity.value:<7}] {f.code}{loc} — {f.message}")
            if f.evidence:
                print(f"        preuve : {f.evidence}")
            if f.hint:
                print(f"        piste  : {f.hint}")
    return EXIT[rep.verdict]


if __name__ == "__main__":
    sys.exit(main())
