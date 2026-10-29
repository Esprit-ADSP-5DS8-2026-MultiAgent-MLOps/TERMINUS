"""Copie de travail isolée + vérification d'empreinte : l'ORIGINAL n'est jamais modifié."""
from __future__ import annotations

import shutil
from pathlib import Path

from validator.contract import ARTIFACT_EXTENSIONS
from validator.fs import walk_project
from validator.handoff import compute_project_hash


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, source: Path, root: Path, meta: Path, source_hash: str, removed: list[str]):
        self.source, self.root, self.meta, self.source_hash, self.removed_artifacts = source, root, meta, source_hash, removed

    @classmethod
    def create(cls, source, dest) -> "Workspace":
        source, dest = Path(source).resolve(), Path(dest).resolve()
        if not source.is_dir():
            raise WorkspaceError(f"{source} n'est pas un dossier")
        if dest == source or dest.is_relative_to(source) or source.is_relative_to(dest):
            raise WorkspaceError("la copie de travail ne doit pas contenir ni être dans le projet source")
        if dest.exists() and any(dest.iterdir()):
            raise WorkspaceError(f"{dest} existe déjà et n'est pas vide : choisis un nouveau dossier par tentative")
        h, _ = compute_project_hash(source)
        files, _links = walk_project(source)
        listed = {f.as_posix() for f in files}
        removed = []
        dest.mkdir(parents=True, exist_ok=True)
        for rel in files:
            if rel.suffix.lower() in ARTIFACT_EXTENSIONS:      # artefacts binaires : jamais copiés (jamais chargés)
                removed.append(rel.as_posix())
                continue
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, tgt)
        meta = dest.with_name(dest.name + ".repair")
        (meta / "patches").mkdir(parents=True, exist_ok=True)
        return cls(source, dest, meta, h, sorted(removed))

    def source_untouched(self) -> bool:
        return compute_project_hash(self.source)[0] == self.source_hash

    def hash(self) -> str:
        return compute_project_hash(self.root)[0]
