"""Passage de relais validateur -> agent 1 : empreinte du projet, copie validée, section `handoff`,
sceau d'intégrité du rapport et vérification côté consommateur.

    python -m validator.handoff schema docs/compatibility_report.schema.json
    python -m validator.handoff verify compatibility_report.json [dossier_projet]
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .contract import ARTIFACT_EXTENSIONS, MODEL_ARTIFACT
from .fs import walk_project
from .models import (REPORT_SCHEMA_VERSION, CarryForward, Confirmation, Handoff, Severity,
                     ValidationReport, Verdict)

SUPPORTED_SCHEMA_MAJOR = REPORT_SCHEMA_VERSION.split(".")[0]
INTERFACES = {
    "train": "train(data_path, output_dir) -> dict[str, float]  (écrit model.joblib dans output_dir)",
    "load_model": "load_model(model_path)",
    "predict": "predict(model, data: dict | list[dict] | DataFrame) -> list  (JSON-sérialisable)",
}


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashable_files(files):
    """Les artefacts binaires commités sont ignorés partout (jamais chargés, ré-entraînés)."""
    return sorted((f for f in files if f.suffix.lower() not in ARTIFACT_EXTENSIONS), key=lambda p: p.as_posix())


def compute_project_hash(root, files=None):
    """Empreinte déterministe du contenu : sha256 sur (chemin, sha256 du fichier), triés. Retourne (hash, nb_fichiers)."""
    root = Path(root)
    if files is None:
        files, _ = walk_project(root)
    h, n = hashlib.sha256(), 0
    for rel in hashable_files(files):
        h.update(f"{rel.as_posix()}\0{_file_sha(root / rel)}\n".encode())
        n += 1
    return h.hexdigest(), n


def _feature_columns(root, rel, target):
    try:
        with open(Path(root) / rel, newline="", encoding="utf-8") as f:
            return [c for c in next(csv.reader(f)) if c != target]
    except Exception:
        return []


def build_handoff(report: ValidationReport, ctx) -> Handoff:
    verdict = report.verdict
    dyn = next((l for l in report.levels if l.name == "dynamic"), None)
    if verdict == Verdict.REJECTED:
        codes = sorted({f.code for f in report.findings if f.severity == Severity.BLOCKER})
        return Handoff(status="BLOCKED", can_proceed=False, requires_human_confirmation=False,
                       reason=f"{len(codes)} type(s) de bloquant : " + ", ".join(codes[:5]))
    if verdict == Verdict.NEEDS_CONFIRMATION:
        msg = next((f.message for f in report.findings if f.severity == Severity.CONFIRM), "confirmation requise")
        return Handoff(status="AWAITING_CONFIRMATION", can_proceed=False, requires_human_confirmation=True, reason=msg)
    if dyn is None or dyn.status != "done":
        return Handoff(status="INCOMPLETE", can_proceed=False, requires_human_confirmation=False,
                       reason="exécution de contrôle non réalisée : " + ((dyn.reason if dyn else None) or "niveau dynamique absent"))

    m = ctx.manifest
    data = dict(ctx.facts.get("data") or {})
    carry = [CarryForward(code=f.code, message=f.message, source="finding")
             for f in report.findings if f.severity == Severity.WARNING]
    balance = data.get("class_balance") or {}
    if len(balance) >= 2 and min(balance.values()) > 0 and max(balance.values()) / min(balance.values()) >= 2:
        ratio = max(balance.values()) / min(balance.values())
        carry.append(CarryForward(code="HANDOFF.IMBALANCED_CLASSES", source="derived",
                                  message=f"Classes déséquilibrées (ratio {ratio:.1f}:1) : préférer roc_auc/f1 à accuracy, split stratifié."))
    if data.get("rows", 10**9) < 2000:
        carry.append(CarryForward(code="HANDOFF.SMALL_DATASET", source="derived",
                                  message=f"{data['rows']} lignes : métriques très variables, calibrer le seuil du quality gate sur plusieurs graines."))
    codes = {f.code for f in report.findings}
    if "C9.NON_DETERMINISTIC" in codes:
        deterministic = False
    else:
        deterministic = True if (ctx.facts.get("dynamic") or {}).get("deterministic") else None
    return Handoff(
        status="READY_WITH_WARNINGS" if report.summary.get("WARNING") else "READY",
        can_proceed=True, requires_human_confirmation=False,
        task=m.project.task, target=m.data.target, primary_metric=m.evaluation.primary_metric,
        min_threshold=m.evaluation.min_threshold, data_path=m.data.path, data_summary=data,
        feature_columns=_feature_columns(ctx.root, m.data.path, m.data.target),
        entrypoints={"main": "main.py", "train": m.entrypoints.train, "predict": m.entrypoints.predict},
        interfaces=INTERFACES, model_artifact=MODEL_ARTIFACT,
        pinned_dependencies={k: v for k, v in (ctx.facts.get("requirements") or {}).items() if v},
        observed_metrics={k: float(x) for k, x in (report.observed_metrics or {}).items()} or None,
        deterministic=deterministic, carry_forward=carry)


def _snapshot(ctx, base, project_hash) -> str:
    dest = Path(base).expanduser().resolve() / project_hash[:12]
    if dest.exists():
        if compute_project_hash(dest)[0] == project_hash:
            return str(dest)
        shutil.rmtree(dest)
    for rel in hashable_files(ctx.files):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ctx.root / rel, target)
        target.chmod(0o444)  # lecture seule : la copie validée ne doit plus changer
    if compute_project_hash(dest)[0] != project_hash:
        raise RuntimeError("copie validée incohérente avec l'empreinte")
    return str(dest)


def _canonical(report: ValidationReport) -> str:
    return json.dumps(report.model_dump(mode="json", exclude={"report_sha256"}), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))


def seal(report: ValidationReport) -> ValidationReport:
    """Sceau d'intégrité : détecte toute modification du rapport (ce n'est PAS une signature : aucun secret)."""
    report.report_sha256 = hashlib.sha256(_canonical(report).encode()).hexdigest()
    return report


def finalize(report: ValidationReport, ctx, *, snapshot_dir=None, confirmed_by=None, confirm_note=None) -> ValidationReport:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report.run_id, report.created_at = uuid.uuid4().hex[:12], now
    if ctx.root.is_dir():
        report.project_hash, report.hashed_files = compute_project_hash(ctx.root, ctx.files)
    if confirmed_by:  # trace seulement : ne change JAMAIS le verdict
        mf = ctx.root / ctx.cfg.manifest_name
        report.confirmation = Confirmation(confirmed_by=confirmed_by, confirmed_at=now, note=confirm_note,
                                           manifest_sha256=_file_sha(mf) if mf.is_file() else None)
    report.handoff = build_handoff(report, ctx)
    if snapshot_dir and report.handoff.can_proceed and report.project_hash:
        try:
            report.snapshot_path = _snapshot(ctx, snapshot_dir, report.project_hash)
        except Exception as e:
            from .models import Finding
            report.findings.append(Finding(rule="C12", code="HANDOFF.SNAPSHOT_FAILED", severity=Severity.INFO, level="handoff",
                                           message=f"Copie validée impossible : {type(e).__name__}: {e}"))
            report.summary["INFO"] = report.summary.get("INFO", 0) + 1
    return seal(report)


@dataclass
class HandoffCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def verify_handoff(report, project_dir=None) -> HandoffCheck:
    """Porte d'entrée de l'agent 1 / de l'orchestrateur. `report` : chemin JSON ou ValidationReport."""
    if not isinstance(report, ValidationReport):
        report = ValidationReport.model_validate_json(Path(report).read_text(encoding="utf-8"))
    why = []
    if report.schema_version.split(".")[0] != SUPPORTED_SCHEMA_MAJOR:
        why.append(f"version de schéma non supportée : {report.schema_version} (attendu {SUPPORTED_SCHEMA_MAJOR}.x)")
    if report.report_sha256 != hashlib.sha256(_canonical(report).encode()).hexdigest():
        why.append("rapport modifié après sa production (sceau invalide)")
    if report.handoff is None or not report.handoff.can_proceed:
        h = report.handoff
        why.append(f"passage de relais refusé : {h.status if h else 'absent'}" + (f" ({h.reason})" if h and h.reason else ""))
    target = project_dir or report.snapshot_path
    if not target:
        why.append("aucun dossier à vérifier (ni project_dir, ni snapshot_path dans le rapport)")
    elif not Path(target).is_dir():
        why.append(f"dossier introuvable : {target}")
    elif report.project_hash and compute_project_hash(target)[0] != report.project_hash:
        why.append("le contenu du projet a changé depuis la validation (empreinte différente)")
    return HandoffCheck(ok=not why, reasons=why)


def schema_json() -> str:
    return json.dumps(ValidationReport.model_json_schema(), indent=2, ensure_ascii=False, sort_keys=True)


def _main(argv) -> int:
    if argv[:1] == ["schema"]:
        if len(argv) > 1:
            Path(argv[1]).write_text(schema_json() + "\n", encoding="utf-8")
            print(f"schéma écrit : {argv[1]}")
        else:
            print(schema_json())
        return 0
    if argv[:1] == ["verify"] and len(argv) >= 2:
        res = verify_handoff(argv[1], argv[2] if len(argv) > 2 else None)
        print("OK : le passage de relais est valide" if res.ok else "REFUSÉ :\n  - " + "\n  - ".join(res.reasons))
        return 0 if res.ok else 30
    print("usage : python -m validator.handoff schema [fichier] | verify rapport.json [dossier_projet]")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
