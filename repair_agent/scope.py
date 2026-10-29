"""Garde de périmètre : protège les choix scientifiques, les données et les tests (règles = données : policy/repair_policy.yaml).

  - chemins protégés (données, tests) : écriture REFUSÉE et tentative signalée → l'agent ne peut pas « réussir » en modifiant le test ;
  - métriques écrites en dur, secrets ajoutés : écriture REFUSÉE et signalée ;
  - choix scientifique (modèle, cible, métrique, découpage) : écriture appliquée mais SIGNALÉE → Judge + approbation humaine.
"""
from __future__ import annotations

import difflib
import fnmatch
import re
from dataclasses import dataclass

from agentic_core.schemas import ScopeFlag
from policy import RepairPolicy, load_repair_policy
from validator.check_static import SECRET_PATTERNS

SEED_KWARG = re.compile(r",?\s*random_state\s*=\s*[\w.]+")


def is_protected(path: str, policy: RepairPolicy) -> bool:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    return any(fnmatch.fnmatch(p, pat) or (pat.startswith("**/") and fnmatch.fnmatch(p, pat[3:])) for pat in policy.protected_paths)


def changed_lines(before: str | None, after: str | None) -> tuple[list[str], list[str]]:
    a, b = (before or "").splitlines(), (after or "").splitlines()
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += a[i1:i2]
        if tag in ("replace", "insert"):
            added += b[j1:j2]
    return removed, added


@dataclass
class ScopeGuard:
    policy: RepairPolicy

    @classmethod
    def default(cls) -> "ScopeGuard":
        return cls(load_repair_policy())

    def check_write(self, rel: str) -> ScopeFlag | None:
        if is_protected(rel, self.policy):
            return ScopeFlag(file=rel, kind="protected_path", blocked=True,
                             detail="chemin protégé (données ou tests du projet) : modification interdite — l'agent ne doit pas réussir en modifiant le test ou les données")
        return None

    def scan_change(self, rel: str, before: str | None, after: str | None) -> list[ScopeFlag]:
        """Analyse un changement de contenu (utilisé à l'écriture par le Repair ET, indépendamment, par le Judge)."""
        if not rel.endswith((".py", ".ipynb", ".yaml", ".yml", ".txt", ".cfg", ".toml")):
            return []
        removed, added = changed_lines(before, after)
        flags: list[ScopeFlag] = []
        hard = re.compile(self.policy.hardcoded_metric_regex, re.I)
        for line in added:
            if hard.search(line):
                flags.append(ScopeFlag(file=rel, kind="hardcoded_metric", blocked=True, detail=f"métrique écrite en dur : {line.strip()[:100]}"))
                break
        for line in added:
            for label, rx in SECRET_PATTERNS:
                if rx.search(line):
                    flags.append(ScopeFlag(file=rel, kind="secret_added", blocked=True, detail=f"secret ajouté ({label})"))
                    break
            else:
                continue
            break
        if rel.endswith(".py"):
            seen: set[str] = set()
            rm_n = {SEED_KWARG.sub("", l).strip() for l in removed}
            ad_n = {SEED_KWARG.sub("", l).strip() for l in added}
            both = rm_n & ad_n                  # même ligne, seule la graine diffère : correction mécanique autorisée
            for line in removed + added:
                if SEED_KWARG.sub("", line).strip() in both:
                    continue
                for pat in self.policy.scientific_patterns:
                    if pat.label not in seen and re.search(pat.regex, line):
                        seen.add(pat.label)
                        flags.append(ScopeFlag(file=rel, kind="scientific_change", detail=f"choix scientifique touché ({pat.label}) : {line.strip()[:100]}"))
        return flags
