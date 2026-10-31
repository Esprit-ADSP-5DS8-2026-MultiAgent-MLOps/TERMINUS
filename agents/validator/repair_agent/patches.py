"""Gestionnaire de patchs : modifications ciblées, diff enregistré par patch, historique, retour arrière."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from agentic_core.schemas import Modification, PatchRecord


@dataclass
class Patch:
    patch_id: str
    file: str
    action: str
    issue_id: str | None
    reason: str
    before: str | None          # None : le fichier n'existait pas
    after: str | None           # None : fichier supprimé
    diff: str
    risk: str = "low"


def unified(rel: str, before: str | None, after: str | None) -> str:
    a = (before or "").splitlines(keepends=True)
    b = (after or "").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, f"a/{rel}" if before is not None else "/dev/null", f"b/{rel}" if after is not None else "/dev/null"))


class PatchManager:
    def __init__(self, root: Path, meta: Path):
        self.root, self.meta = root, meta
        self.applied: list[Patch] = []
        self.history: list[PatchRecord] = []
        self._n = 0

    def apply(self, rel: str, after: str | None, *, issue_id=None, action="replace", reason="", risk="low") -> Patch:
        p = self.root / rel
        before = p.read_text(encoding="utf-8") if p.is_file() else None
        if before == after:
            raise ValueError("aucun changement")
        self._n += 1
        pid = f"p{self._n:04d}"
        patch = Patch(pid, rel, action, issue_id, reason, before, after, unified(rel, before, after), risk)
        if after is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(after, encoding="utf-8")
        self.applied.append(patch)
        (self.meta / "patches").mkdir(parents=True, exist_ok=True)
        (self.meta / "patches" / f"{pid}.diff").write_text(patch.diff, encoding="utf-8")
        self.history.append(PatchRecord(patch_id=pid, files=[rel], issue_id=issue_id, event="applied", diff=patch.diff))
        return patch

    def rollback(self, patch_id: str) -> list[str]:
        """Annule ce patch ET tous les suivants (ordre inverse) : l'état revient exactement à celui d'avant ce patch."""
        ids = [p.patch_id for p in self.applied]
        if patch_id not in ids:
            raise KeyError(f"patch inconnu ou déjà annulé : {patch_id}")
        undone = []
        while self.applied and patch_id in [p.patch_id for p in self.applied]:
            patch = self.applied.pop()
            target = self.root / patch.file
            if patch.before is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(patch.before, encoding="utf-8")
            undone.append(patch.patch_id)
            self.history.append(PatchRecord(patch_id=patch.patch_id, files=[patch.file], issue_id=patch.issue_id, event="rolled_back"))
        return undone

    def rollback_all(self) -> list[str]:
        return self.rollback(self.applied[0].patch_id) if self.applied else []

    def modifications(self) -> list[Modification]:
        return [Modification(patch_id=p.patch_id, file=p.file, issue_id=p.issue_id, action=p.action, reason=p.reason,
                             diff_ref=f"patches/{p.patch_id}.diff", risk=p.risk) for p in self.applied]

    def full_diff(self) -> str:
        return "".join(p.diff for p in self.applied)
