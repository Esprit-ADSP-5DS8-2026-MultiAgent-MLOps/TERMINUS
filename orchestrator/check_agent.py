"""
Vérifier qu'un agent respecte le contrat, AVANT de le brancher dans registry.py.

    python -m orchestrator check-agent agent1 agents.agent1.orchestrator_adapter:run
    python -m orchestrator check-agent agent2 mlflow_agent.orchestrator_adapter:run
    python -m orchestrator check-agent repair repair_agent.orchestrator_adapter:run

Ce que fait la commande :
1. lance le vrai validateur sur le projet de test ;
2. produit les entrées de l'agent avec les stubs des agents précédents (formats exacts du contrat) ;
3. appelle l'agent comme le fera l'orchestrateur, puis vérifie sa réponse : AgentResult valide,
   sorties obligatoires présentes, fichiers conformes à leur schéma, écrits dans le workdir,
   projet laissé intact.

Chaque membre la lance sur sa machine : quand elle affiche « conforme », l'intégration se résume
à une ligne dans registry.REAL.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from common.schemas import (
    OUTPUT_SCHEMAS,
    OUTPUTS,
    SCHEMA_VERSION,
    AgentInput,
    AgentResult,
    AgentStatus,
)

from .nodes import run_validator
from .registry import STUBS, _load

ROOT = Path(__file__).resolve().parents[1]

# Projet de test par défaut : un projet accepté, sauf pour les agents qui traitent un refus.
DEFAULT_PROJECT = {
    "propose_manifest": "eval/projects/variants/09_no_manifest",
    "repair": "eval/projects/variants/02_unpinned_deps",
}
# Agents qui tournent avant chacun (leurs stubs fournissent les entrées).
UPSTREAM = {
    "agent1": [],
    "propose_manifest": [],
    "repair": [],
    "agent2": ["agent1"],
    "agent3": ["agent1", "agent2"],
    "agent4": ["agent1", "agent2", "agent3"],
}
EXPECTED_AGENT = {"propose_manifest": "agent1"}


@dataclass
class Report:
    checks: list[tuple[bool, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self, ok: bool, text: str) -> bool:
        self.checks.append((bool(ok), text))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for ok, _ in self.checks)


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def prepare(name: str, project: Path, out: Path) -> AgentInput:
    """Valide le projet et fabrique les entrées de l'agent `name` avec les stubs en amont."""
    if out.exists():
        shutil.rmtree(out)
    (out / "validator").mkdir(parents=True)
    report = out / "validator" / "compatibility_report.json"
    proc = run_validator(str(project), report, out / "snapshot")
    if not report.exists():
        raise RuntimeError(f"le validateur n'a pas produit de rapport (code {proc.returncode}) : "
                           f"{proc.stderr.strip()[-300:]}")
    data = json.loads(report.read_text(encoding="utf-8"))
    status = data["handoff"]["status"]
    if name in ("agent1", "agent2", "agent3", "agent4") and not data["handoff"]["can_proceed"]:
        raise RuntimeError(f"projet de test refusé par le validateur ({status}) : choisir un projet accepté")
    snapshot = data.get("snapshot_path") or str(project)
    inputs = {"compatibility_report": str(report)}
    for upstream in UPSTREAM[name]:
        wd = out / upstream
        wd.mkdir()
        result = STUBS[upstream](AgentInput(run_id="check", project_dir=snapshot, report_path=str(report),
                                            workdir=str(wd), inputs=dict(inputs)))
        inputs.update(result.outputs)
    workdir = out / name
    workdir.mkdir()
    return AgentInput(run_id="check", project_dir=snapshot, report_path=str(report),
                      workdir=str(workdir), inputs=inputs)


def check(name: str, target, project: Path, out: Path) -> Report:
    """target : "module:fonction" ou directement la fonction."""
    rep = Report()
    fn = _load(target) if isinstance(target, str) else target
    inp = prepare(name, project, out)
    rep.notes.append(f"entrées fournies : {', '.join(sorted(inp.inputs))}")
    before = _tree_hash(Path(inp.project_dir))

    started = time.monotonic()
    try:
        raw = fn(inp)
    except Exception as exc:  # noqa: BLE001 - tout échec de l'agent est un résultat de la vérification
        rep.check(False, f"l'agent a levé une exception : {type(exc).__name__}: {exc}"[:400])
        return rep
    rep.notes.append(f"durée : {time.monotonic() - started:.1f} s")

    try:
        result = AgentResult.model_validate(raw.model_dump() if hasattr(raw, "model_dump") else raw)
    except Exception as exc:  # noqa: BLE001 - tout échec de l'agent est un résultat de la vérification
        rep.check(False, f"la réponse n'est pas un AgentResult valide : {exc}"[:400])
        return rep
    rep.check(True, f"AgentResult valide (statut {result.status.value}, can_proceed={result.can_proceed})")

    expected = EXPECTED_AGENT.get(name, name)
    rep.check(result.agent == expected, f"champ agent = « {expected} » (reçu « {result.agent} »)")
    rep.check(result.schema_version == SCHEMA_VERSION,
              f"schema_version = {SCHEMA_VERSION} (reçu {result.schema_version})")
    if result.status in (AgentStatus.FAILED, AgentStatus.NEEDS_APPROVAL):
        rep.check(bool(result.reasons), f"statut {result.status.value} expliqué dans reasons")
    if result.status == AgentStatus.FAILED:
        rep.check(not result.can_proceed, "FAILED implique can_proceed=False")
    else:
        missing = [o for o in OUTPUTS.get(name, []) if o not in result.outputs]
        rep.check(not missing, "sorties obligatoires présentes"
                  + (f" (manquantes : {', '.join(missing)})" if missing else
                     f" ({', '.join(OUTPUTS.get(name, [])) or 'aucune'})"))

    workdir = Path(inp.workdir).resolve()
    for key, path in result.outputs.items():
        p = Path(path)
        if not rep.check(p.exists(), f"{key} : fichier présent ({path})"):
            continue
        rep.check(p.resolve().is_relative_to(workdir), f"{key} : écrit dans le workdir de l'agent")
        schema = OUTPUT_SCHEMAS.get(key)
        if schema is None:
            continue
        try:
            obj = schema.model_validate_json(p.read_text(encoding="utf-8"))
            rep.check(True, f"{key} : conforme à {schema.__name__}")
            version = getattr(obj, "schema_version", SCHEMA_VERSION)
            if version != SCHEMA_VERSION:
                rep.notes.append(f"{key} : schema_version {version} (contrat actuel : {SCHEMA_VERSION})")
        except Exception as exc:  # noqa: BLE001 - tout échec de l'agent est un résultat de la vérification
            rep.check(False, f"{key} : non conforme à {schema.__name__} : {str(exc)[:300]}")

    rep.check(_tree_hash(Path(inp.project_dir)) == before, "projet d'entrée laissé intact (lecture seule)")
    return rep


def main(name: str, target: str, project: str | None, out: str) -> int:
    if name not in UPSTREAM:
        print(f"Agent inconnu : {name} (choix : {', '.join(UPSTREAM)})")
        return 2
    project_dir = (ROOT / (project or DEFAULT_PROJECT.get(name, "eval/projects/credit_risk"))).resolve()
    out_dir = Path(out).resolve() / name
    print(f"Vérification de {target} comme « {name} » (contrat {SCHEMA_VERSION})")
    print(f"Projet de test : {project_dir}")
    try:
        rep = check(name, target, project_dir, out_dir)
    except Exception as exc:  # noqa: BLE001 - tout échec de l'agent est un résultat de la vérification
        print(f"✖ impossible de lancer la vérification : {type(exc).__name__}: {exc}")
        return 2
    for ok, text in rep.checks:
        print(f"  {'✔' if ok else '✖'} {text}")
    for note in rep.notes:
        print(f"  · {note}")
    print(f"Fichiers : {out_dir}")
    print("Conforme : il reste à l'ajouter dans orchestrator/registry.py (REAL)." if rep.ok
          else "Non conforme : corriger l'adaptateur puis relancer.")
    return 0 if rep.ok else 1
