from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .model import AdmissionContract, RepairPolicy

DIR = Path(__file__).parent


def list_contracts() -> list[str]:
    return sorted(p.stem.replace("_", "-") for p in (DIR / "contracts").glob("*.yaml"))


def load_contract(name_or_path: str | Path = "sklearn-tabular") -> AdmissionContract:
    """Par nom de profil (sklearn-tabular, training-only…) ou par chemin d'un fichier YAML. Ajouter un profil = ajouter un fichier."""
    p = Path(str(name_or_path))
    if not p.suffix:
        p = DIR / "contracts" / (str(name_or_path).replace("-", "_") + ".yaml")
    if not p.is_file():
        raise FileNotFoundError(f"contrat introuvable : {name_or_path} (profils : {list_contracts()})")
    return AdmissionContract.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def load_repair_policy() -> RepairPolicy:
    return RepairPolicy.model_validate(yaml.safe_load((DIR / "repair_policy.yaml").read_text(encoding="utf-8")))


def render_contract_for_prompt(c: AdmissionContract, check_ids: set[str] | None = None) -> str:
    """Le contrat, en données, injecté dans le prompt du Validator (le LLM ne décide pas du statut)."""
    rows = [k for k in c.checks if check_ids is None or k.id in check_ids]
    lines = [f"PROFIL : {c.id} v{c.version} — {c.description}",
             f"Frameworks dans le périmètre : {', '.join(c.frameworks_in_scope) or 'non restreint'}",
             f"Hors périmètre : {', '.join(c.frameworks_out_of_scope) or '—'}",
             "CATALOGUE DE CONTRÔLES (rapporte chaque id ; M = obligatoire, R = recommandé) :"]
    for k in rows:
        na = "".join(f"\n      NOT_APPLICABLE autorisé si « {n.id} » : {n.description}" for n in k.not_applicable_when)
        lines.append(f"  - {k.id} [{k.level}] ({'M' if k.requirement == 'mandatory' else 'R'}) {k.title} : {k.description}{na}")
    return "\n".join(lines)
