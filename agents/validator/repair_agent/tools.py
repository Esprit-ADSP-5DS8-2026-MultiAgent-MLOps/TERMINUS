"""Outils du Repair : lecture libre, ÉCRITURE CONFINÉE à la copie de travail, gardée par le périmètre."""
from __future__ import annotations

from agentic_core.facts import static_facts
from agentic_core.sandbox_exec import SandboxRunner
from agentic_core.schemas import ScopeFlag
from agentic_core.toolkit import BaseToolBox, ToolError, fn_spec, j

from .patches import PatchManager
from .runner import run_checks
from .scope import ScopeGuard
from .workspace import Workspace

MAX_WRITE_CHARS = 60_000


class RepairToolBox(BaseToolBox):
    def __init__(self, ws: Workspace, patches: PatchManager, guard: ScopeGuard, runner: SandboxRunner | None):
        super().__init__(ws.root)                       # lecture ET écriture : uniquement dans la copie de travail
        self.ws, self.patches, self.guard, self.runner = ws, patches, guard, runner
        self.flags: list[ScopeFlag] = []
        self._flag_owner: dict[int, str] = {}          # id(flag) -> patch_id qui l'a provoqué (signalements non bloquants)
        self.last_checks: list = []
        self.tools.update(analyze_code=self.analyze_code, replace_in_file=self.replace_in_file, write_file=self.write_file,
                          delete_file=self.delete_file, revert_patch=self.revert_patch, run_checks=self.run_checks)

    def specs(self):
        return self.common_specs() + [
            fn_spec("analyze_code", "Faits statiques sur la copie de travail (syntaxe, imports, appels dangereux, chemins, graines…). Utile pour vérifier ta correction.", {}),
            fn_spec("replace_in_file", "Remplace UNE occurrence exacte de old_str par new_str dans un fichier de la copie de travail (modification ciblée, préférée). "
                    "old_str doit exister une seule fois, copié SANS les numéros « N| ».",
                    {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}, "issue_id": {"type": "string"}, "reason": {"type": "string"}},
                    ["path", "old_str", "new_str", "issue_id", "reason"]),
            fn_spec("write_file", "Crée ou réécrit un fichier entier (pour un fichier absent : requirements.txt, mlops.yaml…). Données et tests protégés : refusé.",
                    {"path": {"type": "string"}, "content": {"type": "string"}, "issue_id": {"type": "string"}, "reason": {"type": "string"}},
                    ["path", "content", "issue_id", "reason"]),
            fn_spec("delete_file", "Supprime un fichier de la copie de travail (ex. artefact). Données et tests protégés : refusé.",
                    {"path": {"type": "string"}, "issue_id": {"type": "string"}, "reason": {"type": "string"}}, ["path", "issue_id", "reason"]),
            fn_spec("revert_patch", "Annule un patch (et les suivants) : retour arrière.", {"patch_id": {"type": "string"}}, ["patch_id"]),
            fn_spec("run_checks", "Lance les vérifications sur la copie corrigée (syntaxe, analyse statique, dépendances, import, tests unitaires du projet).", {}),
            fn_spec("submit_repair", "Dépose le compte rendu FINAL (une seule fois). Les modifications réelles sont lues dans l'historique des patchs, pas dans ce compte rendu.", {
                "results": {"type": "array", "description": "UN élément par problème reçu", "items": {"type": "object", "properties": {
                    "issue_id": {"type": "string"}, "outcome": {"type": "string", "enum": ["fixed", "not_fixed", "needs_human"]},
                    "patch_ids": {"type": "array", "items": {"type": "string"}}, "explanation": {"type": "string"}}, "required": ["issue_id", "outcome", "explanation"]}},
                "summary": {"type": "string"}}, ["results"]),
        ]

    def analyze_code(self) -> str:
        return j({k: v for k, v in static_facts(self.root).items() if not k.startswith("_")})

    # ------------------------------------------------------------------ écriture gardée
    def _commit(self, path, after, issue_id, action, reason):
        rel = self.rel(self.resolve(path))
        flag = self.guard.check_write(rel)
        if flag is None and after is not None:
            before = (self.root / rel).read_text(encoding="utf-8") if (self.root / rel).is_file() else None
            blocked = [f for f in self.guard.scan_change(rel, before, after) if f.blocked]
            if blocked:
                flag = blocked[0]
        if flag:
            self.flags.append(flag)
            raise ToolError(f"écriture REFUSÉE et signalée au Judge : {flag.detail}")
        before = (self.root / rel).read_text(encoding="utf-8") if (self.root / rel).is_file() else None
        scientific = [f for f in self.guard.scan_change(rel, before, after) if f.kind == "scientific_change"] if after is not None else []
        try:
            patch = self.patches.apply(rel, after, issue_id=issue_id, action=action, reason=str(reason)[:300], risk="scientific" if scientific else "low")
        except ValueError as e:
            raise ToolError(str(e))
        self.flags += scientific
        for f in scientific:
            self._flag_owner[id(f)] = patch.patch_id
        out = {"ok": True, "patch_id": patch.patch_id, "file": rel}
        if scientific:
            out["warning"] = "choix scientifique touché : la modification est signalée au Judge et exigera l'approbation d'un humain. Si possible, reviens en arrière (revert_patch)."
        return j(out)

    def replace_in_file(self, path, old_str, new_str, issue_id, reason):
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"fichier introuvable : {path}")
        text = p.read_text(encoding="utf-8")
        if text.count(old_str) != 1:
            raise ToolError(f"old_str doit apparaître exactement une fois (trouvé {text.count(old_str)} fois)")
        return self._commit(path, text.replace(old_str, new_str), issue_id, "replace", reason)

    def write_file(self, path, content, issue_id, reason):
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError("contenu trop volumineux")
        existed = self.resolve(path).is_file()
        return self._commit(path, content, issue_id, "rewrite" if existed else "create", reason)

    def delete_file(self, path, issue_id, reason):
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"fichier introuvable : {path}")
        return self._commit(path, None, issue_id, "delete", reason)

    def revert_patch(self, patch_id):
        try:
            undone = self.patches.rollback(patch_id)
        except KeyError as e:
            raise ToolError(str(e))
        # seuls les signalements des patchs réellement annulés disparaissent ; les tentatives bloquées et les patchs encore appliqués restent signalés
        self.flags = [f for f in self.flags if f.blocked or self._flag_owner.get(id(f)) not in set(undone)]
        return j({"ok": True, "rolled_back": undone})

    def run_checks(self) -> str:
        self.last_checks = run_checks(self.root, self.runner)
        return j([t.model_dump(exclude={"command"}) for t in self.last_checks])
