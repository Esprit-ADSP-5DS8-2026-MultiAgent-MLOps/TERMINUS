from __future__ import annotations

import os
from pathlib import Path

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}


def walk_project(root: Path):
    """Retourne (fichiers, liens symboliques) en chemins relatifs, sans suivre les liens."""
    files, links = [], []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        keep = []
        for d in dirnames:
            full = Path(dirpath) / d
            if d in IGNORED_DIRS:
                continue
            if full.is_symlink():
                links.append(full.relative_to(root))
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            full = Path(dirpath) / f
            rel = full.relative_to(root)
            (links if full.is_symlink() else files).append(rel)
    return sorted(files), sorted(links)


def read_text_safe(path: Path, limit: int = 1_000_000):
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
