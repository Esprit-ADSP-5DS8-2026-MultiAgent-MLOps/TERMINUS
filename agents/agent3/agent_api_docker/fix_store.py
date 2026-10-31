"""Memoire durable des corrections : un correctif valide n'est plus perdu a la regeneration suivante.

Probleme corrige : le LLM patchait `workspaces/<id>/generated/`, mais chaque execution efface ce dossier et
regenere tout depuis les templates -> la meme erreur revenait a chaque fois.

Principe :
  - quand une execution REUSSIT, les patchs appliques (et verifies) sont ecrits dans `fixes/<project_id>.json`
    (dossier hors de workspaces/, a versionner dans git) ;
  - a l'execution suivante, juste apres la generation, ils sont REJOUES dans l'ordre, sans appel au LLM ;
  - un patch devenu inapplicable (template ou rapport modifie : `old` introuvable) est ignore, signale,
    puis retire de la memoire a la prochaine reussite ; la boucle de correction reprend alors la main ;
  - un echec n'ecrit JAMAIS rien : la memoire ne contient que des corrections qui ont mene a un succes.
Seuls les fichiers generes (fixer.FIXABLE_FILES) sont concernes : le code du projet n'est jamais modifie.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import config
from .fixer import FIXABLE_FILES, Patch
from .tools.files import patch_file

STORE_VERSION = 1


def store_path(project_id: str, fixes_dir: Optional[Path] = None) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in project_id)
    return Path(fixes_dir or config.FIXES_DIR) / f"{safe}.json"


def load(project_id: str, fixes_dir: Optional[Path] = None) -> list[Patch]:
    """Patchs memorises (ordre d'application). Fichier absent ou illisible -> liste vide, jamais d'exception."""
    path = store_path(project_id, fixes_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for p in data.get("patches", []):
            if isinstance(p, dict) and {"path", "old", "new"} <= set(p):
                out.append(Patch(str(p["path"]), str(p["old"]), str(p["new"])))
        return out
    except (OSError, ValueError, AttributeError):
        return []


def merge(*groups: list[Patch]) -> list[Patch]:
    """Concatene en gardant l'ordre et en supprimant les doublons exacts."""
    seen: set[str] = set()
    out: list[Patch] = []
    for g in groups:
        for p in g:
            if p.key() not in seen:
                seen.add(p.key())
                out.append(p)
    return out


def replay(generated_dir: Path, patches: list[Patch]) -> tuple[list[Patch], list[str]]:
    """Rejoue les patchs sur les fichiers qui viennent d'etre regeneres.

    Retourne (patchs rejoues, raisons des patchs ignores). Ne leve jamais : un patch obsolete est signale, pas fatal.
    """
    applied: list[Patch] = []
    skipped: list[str] = []
    for p in patches:
        if p.path not in FIXABLE_FILES:
            skipped.append(f"{p.path} : fichier non modifiable")
        elif not (Path(generated_dir) / p.path).is_file():
            skipped.append(f"{p.path} : fichier absent des fichiers generes")
        else:
            try:
                patch_file(Path(generated_dir), p.path, p.old, p.new)
            except ValueError:
                skipped.append(f"{p.path} : texte a remplacer introuvable ({p.old.strip()[:60]!r}...) : template ou rapport modifie ?")
            else:
                applied.append(p)
    return applied, skipped


def verified_for_storage(new_patches: list[Patch], api_ok: bool, docker_ok: bool) -> list[Patch]:
    """Patchs assez verifies pour etre memorises.

    - api.py : verifie par la validation + les tests locaux (api_ok) ;
    - Dockerfile / requirements / .dockerignore / compose : seulement verifies par un vrai build + tests dans le conteneur.
    """
    return [p for p in new_patches if docker_ok or (api_ok and p.path == "api.py")]


def save(project_id: str, patches: list[Patch], fixes_dir: Optional[Path] = None, **meta) -> Path:
    """Ecriture atomique (fichier temporaire puis remplacement) : jamais de fichier a moitie ecrit."""
    path = store_path(project_id, fixes_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": STORE_VERSION, "project_id": project_id, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **meta, "patches": [{"path": p.path, "old": p.old, "new": p.new} for p in patches]}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
