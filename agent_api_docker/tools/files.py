"""Ecriture sure dans le workspace : refuse de sortir du dossier de travail."""
from __future__ import annotations

from pathlib import Path


def write_file(workspace: Path, rel_path: str, content: str) -> Path:
    target = (workspace / rel_path).resolve()
    if workspace.resolve() not in target.parents:
        raise ValueError(f"Chemin hors workspace refuse : {rel_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def patch_file(workspace: Path, rel_path: str, old: str, new: str) -> None:
    path = (workspace / rel_path).resolve()
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError(f"'old' doit apparaitre exactement une fois dans {rel_path}")
    write_file(workspace, rel_path, text.replace(old, new))
