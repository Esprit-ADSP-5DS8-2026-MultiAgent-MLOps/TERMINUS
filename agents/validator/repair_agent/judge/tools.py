"""Outils du Judge : LECTURE SEULE. Il ne croit pas le rapport du Repair : diff et signalements de périmètre sont RECALCULÉS
à partir de l'original et de la copie corrigée."""
from __future__ import annotations

import hashlib
from pathlib import Path

from agentic_core.schemas import RepairReport, RunStatus, ScopeFlag, ValidatorReport
from agentic_core.toolkit import BaseToolBox, ToolError, fn_spec, j
from repair_agent.patches import unified
from repair_agent.scope import ScopeGuard
from validator.fs import walk_project


def _read(p: Path):
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


class JudgeToolBox(BaseToolBox):
    def __init__(self, source: Path, repaired: Path, repair: RepairReport, validator: ValidatorReport | None = None, guard: ScopeGuard | None = None):
        super().__init__(repaired)                       # `root` = copie corrigée ; l'original est lu via `source`
        self.source = Path(source).resolve()
        self.repair, self.validator, self.guard = repair, validator, guard or ScopeGuard.default()
        self.diff_examined = False
        self.changes, self.flags, self.real_diff = self._compute()
        self.tools.update(list_changes=self.list_changes, get_diff=self.get_diff, read_original=self.read_original,
                          read_repaired=self.read_repaired, get_test_results=self.get_test_results,
                          get_metrics_comparison=self.get_metrics_comparison, get_validator_issues=self.get_validator_issues)
        self.tools.pop("read_file")

    # ------------------------------------------------------------------ recalcul indépendant
    def _compute(self):
        def digest(root):
            files, _ = walk_project(root)
            return {f.as_posix(): hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
        a, b = digest(self.source), digest(self.root)
        changes, flags, diffs = [], [], []
        for rel in sorted(set(a) | set(b)):
            if a.get(rel) == b.get(rel):
                continue
            kind = "added" if rel not in a else "removed" if rel not in b else "modified"
            changes.append({"file": rel, "change": kind})
            if kind == "removed" and rel in set(self.repair.removed_artifacts):
                continue                      # artefact binaire non copié par le Repair : déclaré dans `removed_artifacts`
            before = _read(self.source / rel) if rel in a else None
            after = _read(self.root / rel) if rel in b else None
            if (before is None and rel in a) or (after is None and rel in b):
                flags.append(ScopeFlag(file=rel, kind="scientific_change", detail="fichier binaire ou non texte modifié/supprimé : non relisible"))
                continue
            pf = self.guard.check_write(rel)
            if pf:
                flags.append(pf.model_copy(update={"blocked": False, "detail": "MODIFIÉ malgré la protection : " + pf.detail}))
            flags += [f.model_copy(update={"blocked": False}) for f in self.guard.scan_change(rel, before, after)]
            diffs.append(unified(rel, before, after))
        return changes, flags, "".join(diffs)

    @property
    def diff_matches_report(self) -> bool:
        real = [c for c in self.changes if not c["file"].endswith(tuple(self.repair.removed_artifacts))]
        claimed = {m.file for m in self.repair.modifications if m.patch_id != "artifact"}
        return {c["file"] for c in real} == claimed

    def blockers_for_accept(self) -> list[str]:
        """Raisons pour lesquelles un ACCEPT est interdit (protocole, pas jugement)."""
        r, out = self.repair, []
        if r.failure or r.run_status != RunStatus.OK or r.status != "REPAIRED":
            out.append(f"le Repair n'est pas en succès complet (statut {r.status}, run_status {getattr(r.run_status, 'value', r.run_status)}"
                       + (f", échec : {r.failure.kind.value}" if r.failure else "") + ")")
        if self.flags:
            out.append(f"périmètre : {len(self.flags)} signalement(s) indépendant(s) ({', '.join(sorted({f.kind for f in self.flags}))}) → approbation humaine requise")
        if r.requires_human_approval:
            out.append("le rapport de réparation exige une approbation humaine")
        if any(t.status in ("FAIL", "ERROR") for t in r.tests):
            out.append("au moins un test échoue : " + ", ".join(t.name for t in r.tests if t.status in ("FAIL", "ERROR")))
        if not any(t.status == "PASS" for t in r.tests):
            out.append("aucun test réussi : aucune preuve d'exécution")
        if r.metrics_comparison and r.metrics_comparison.behavior_changed:
            out.append("les métriques avant/après diffèrent au-delà de la tolérance : le comportement a changé")
        if not self.diff_matches_report:
            out.append("le diff réel ne correspond pas aux modifications déclarées par le Repair")
        if r.unresolved_issues:
            out.append(f"{len(r.unresolved_issues)} problème(s) non résolu(s)")
        return out

    # ------------------------------------------------------------------ outils
    def evidence_hints(self) -> list[str]:
        """Extraits EXACTS présents dans les sorties d'outils déjà vues (aide la recopie ; chacun est vérifié par is_grounded)."""
        diff = [l[:80] for l in self.real_diff.splitlines() if l[:1] in ("+", "-") and not l.startswith(("+++", "---")) and l[1:].strip()]
        tests = [f'"name": "{t.name}", "status": "{t.status}"' for t in self.repair.tests]
        return [h for h in diff if self.is_grounded(h)][:2] + [h for h in tests if self.is_grounded(h)][:3]

    def specs(self):
        return [
            fn_spec("list_files", "Liste les fichiers de la copie corrigée (chemin, taille) : pour vérifier qu'un chemin ajouté existe.", {}),
            fn_spec("list_changes", "Fichiers réellement modifiés (recalculés : original vs copie corrigée), signalements de périmètre INDÉPENDANTS, "
                    "et ce que le Repair a déclaré.", {}),
            fn_spec("get_diff", "Diff unifié réel d'un fichier (ou de tous si path omis).", {"path": {"type": "string"}}),
            fn_spec("read_original", "Lit un fichier de l'ORIGINAL (lecture seule, non fiable).", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}}, ["path"]),
            fn_spec("read_repaired", "Lit un fichier de la copie corrigée (lecture seule, non fiable).", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}}, ["path"]),
            fn_spec("get_test_results", "Tests relancés par l'infrastructure sur la copie corrigée (nom, statut, commande, code de sortie, extrait).", {}),
            fn_spec("get_metrics_comparison", "Comparaison des métriques avant/après (comportement modifié ?).", {}),
            fn_spec("get_validator_issues", "Problèmes que la réparation devait corriger (rapport du Validator) et ceux restés non résolus.", {}),
            fn_spec("submit_verdict", "Dépose ton verdict FINAL sur CETTE réparation (une seule fois). Jamais une décision d'admission.", {
                "verdict": {"type": "string", "enum": ["ACCEPT", "REVISE", "REJECT"]},
                "criteria": {"type": "array", "description": "EXACTEMENT 5 : correctness, completeness, regression_risk, evidence_quality, scope_compliance", "items": {"type": "object", "properties": {
                    "name": {"type": "string", "enum": ["correctness", "completeness", "regression_risk", "evidence_quality", "scope_compliance"]},
                    "result": {"type": "string", "enum": ["PASS", "CONCERN", "FAIL"]}, "rationale": {"type": "string"},
                    "evidence": {"type": "string", "description": "extrait EXACT d'un diff ou d'une sortie de test, copié depuis un outil"}},
                    "required": ["name", "result", "rationale", "evidence"]}},
                "evidence_examined": {"type": "array", "items": {"type": "string"}, "description": "ce que tu as examiné (fichiers, diffs, tests)"},
                "remaining_concerns": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "next_action": {"type": "string", "enum": ["REVALIDATE", "RETRY", "ESCALATE", "ROLLBACK"]}},
                ["verdict", "criteria", "reason", "next_action"]),
        ]

    def list_changes(self) -> str:
        self.diff_examined = True
        return j({"real_changes": self.changes, "independent_scope_flags": [f.model_dump() for f in self.flags],
                  "repair_declared": [{"file": m.file, "issue_id": m.issue_id, "action": m.action, "risk": m.risk, "reason": m.reason} for m in self.repair.modifications],
                  "repair_scope_flags_during_run": [f.model_dump() for f in self.repair.scope_flags],
                  "diff_matches_declared_modifications": self.diff_matches_report, "removed_artifacts": self.repair.removed_artifacts})

    def get_diff(self, path: str | None = None) -> str:
        self.diff_examined = True
        if not path:
            return j({"diff": self.real_diff or "(aucun changement de texte)"})
        if path not in {c["file"] for c in self.changes}:
            raise ToolError(f"{path} n'a pas été modifié")
        before, after = _read(self.source / path), _read(self.root / path)
        return j({"file": path, "diff": unified(path, before, after)})

    def _view(self, root, path, start_line=1, max_lines=200):
        p = self.resolve(path, root)
        return self.render_file(p, self.rel(p, root), start_line, max_lines)[0]

    def read_original(self, path, start_line=1, max_lines=200):
        return self._view(self.source, path, start_line, max_lines)

    def read_repaired(self, path, start_line=1, max_lines=200):
        return self._view(self.root, path, start_line, max_lines)

    def get_test_results(self) -> str:
        return j([t.model_dump() for t in self.repair.tests])

    def get_metrics_comparison(self) -> str:
        return j(self.repair.metrics_comparison.model_dump() if self.repair.metrics_comparison else {"status": "NOT_AVAILABLE"})

    def get_validator_issues(self) -> str:
        if not self.validator:
            return j({"note": "rapport du Validator non fourni"})
        return j({"validator_status": self.validator.status.value,
                  "issues": [{"issue_id": i.issue_id, "check_id": i.check_id, "severity": i.severity.value, "nature": i.nature.value, "file": i.file, "description": i.description} for i in self.validator.issues],
                  "unresolved_by_repair": [u.model_dump() for u in self.repair.unresolved_issues]})
