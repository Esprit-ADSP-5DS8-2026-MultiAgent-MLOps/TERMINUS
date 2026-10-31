"""Outils de recherche en LECTURE SEULE, confines a des dossiers racines.

Le LLM choisit quoi explorer ; ce module garantit qu'il ne peut ni sortir des racines autorisees,
ni modifier quoi que ce soit, ni noyer son contexte (tailles et nombres de resultats plafonnes).
"""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .. import config
from ..schemas import AnalysisReport

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", ".idea", ".vscode"}
MAX_WALK_ENTRIES = 20_000     # garde-fou : un `find` ne parcourt jamais plus que ca
MAX_JSON_BYTES = 5_000_000


class SandboxError(Exception):
    """Chemin refuse (hors racines autorisees) ou introuvable."""


class Sandbox:
    def __init__(self, roots: Optional[list[Path]] = None):
        self.roots = [Path(r).resolve() for r in (roots or [config.REPO_ROOT, config.WORKSPACES_DIR])]
        self.base = self.roots[0]  # les chemins relatifs sont lus depuis la 1re racine (le depot)

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        p = (p if p.is_absolute() else self.base / p).resolve()
        if not any(p == r or r in p.parents for r in self.roots):
            raise SandboxError(f"Chemin hors des dossiers autorises : {path}")
        return p

    def rel(self, path: Path) -> str:
        for r in self.roots:
            try:
                return path.relative_to(r).as_posix() or "."
            except ValueError:
                continue
        return str(path)

    # ------------------------------------------------------------------ outils
    def list_dir(self, path: str = ".", max_entries: int = 60) -> str:
        d = self.resolve(path)
        if not d.is_dir():
            raise SandboxError(f"Pas un dossier : {path}")
        entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        entries = [e for e in entries if e.name not in SKIP_DIRS]
        lines = [f"{e.name}/" if e.is_dir() else f"{e.name}  ({e.stat().st_size} o)" for e in entries[:max_entries]]
        if len(entries) > max_entries:
            lines.append(f"... {len(entries) - max_entries} autres entrees non affichees")
        return f"{self.rel(d)}/\n" + ("\n".join("  " + x for x in lines) if lines else "  (vide)")

    def find_files(self, pattern: str, root: str = ".", max_results: int = 25, max_depth: int = 8) -> str:
        """Cherche par motif glob sur le NOM du fichier (ex: '*analysis_report*.json')."""
        start = self.resolve(root)
        if not start.is_dir():
            raise SandboxError(f"Pas un dossier : {root}")
        hits: list[str] = []
        seen = 0
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            depth = len(Path(dirpath).relative_to(start).parts)
            if depth >= max_depth:
                dirnames[:] = []
            for name in sorted(filenames):
                seen += 1
                if fnmatch.fnmatch(name.lower(), pattern.lower()):
                    hits.append(self.rel(Path(dirpath) / name))
            if seen > MAX_WALK_ENTRIES or len(hits) >= max_results:
                break
        if not hits:
            return f"Aucun fichier ne correspond a '{pattern}' sous {self.rel(start)}/ (profondeur max {max_depth})."
        extra = " (liste tronquee)" if len(hits) >= max_results else ""
        return f"{len(hits)} resultat(s) pour '{pattern}'{extra} :\n" + "\n".join("  " + h for h in hits[:max_results])

    def inspect_json(self, path: str) -> str:
        """Dit a quel contrat correspond un JSON (rapport d'analyse, rapport de structure, autre)."""
        f = self.resolve(path)
        if not f.is_file():
            raise SandboxError(f"Fichier introuvable : {path}")
        if f.stat().st_size > MAX_JSON_BYTES:
            return f"{self.rel(f)} : fichier trop gros ({f.stat().st_size} o), ignore."
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return f"{self.rel(f)} : JSON illisible ({e})"
        if not isinstance(data, dict):
            return f"{self.rel(f)} : JSON de type {type(data).__name__}, un objet etait attendu."
        kind, detail = classify_report(data)
        keys = ", ".join(list(data)[:14])
        return (f"{self.rel(f)}\n  type detecte : {kind}\n  project_name : {data.get('project_name')}\n"
                f"  schema_version : {data.get('schema_version')}\n  cles de premier niveau : {keys}\n  {detail}")


def classify_report(data: dict) -> tuple[str, str]:
    """('analysis_report'|'structure_report'|'inconnu', detail). Verdict fait par le code, pas par le LLM."""
    try:
        AnalysisReport.model_validate(data)
        return "analysis_report", "Conforme au contrat attendu par l'agent API/Docker."
    except ValidationError as e:
        errs = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:4]]
        if {"ml", "io", "inference"} <= set(data):
            return "analysis_report", "Ressemble a un rapport d'analyse mais NON conforme : " + " | ".join(errs)
        if {"files", "layout"} <= set(data) or "docker_readiness" in data:
            return "structure_report", ("Rapport de STRUCTURE (liste de fichiers, layout) : ce n'est PAS le rapport "
                                        "attendu (il manque ml / io / inference).")
        return "inconnu", "Ne correspond a aucun rapport connu : " + " | ".join(errs)


def norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())
