"""Ecriture sure dans le workspace : refuse de sortir du dossier de travail."""
from __future__ import annotations

import shutil
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


def sync_files(src: Path, dst: Path) -> int:
    """Copie les fichiers de la racine de src vers dst. Sans effet (0) si src et dst sont le MEME dossier.

    Sert a republier les fichiers generes apres un patch ; quand generated/ est aussi le dossier de build,
    il n'y a rien a copier (et copier un fichier sur lui-meme leverait SameFileError).
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        return 0
    n = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            n += 1
    return n
