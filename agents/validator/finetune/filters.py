"""Filtres de qualité des exécutions : fonctions PURES sur un dictionnaire `run` (voir collect.py pour le format).

Chaque filtre renvoie une liste de raisons d'échec (vide = ok). Un exemple d'entraînement doit passer TOUS les filtres, sinon le modèle
apprendrait à « satisfaire les règles » (ex. changer une constante inutilisée) au lieu de réparer pour de vrai.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

OK_STATUSES = ("ACCEPTED", "ACCEPTED_WITH_WARNINGS")
PATH_LITERAL = re.compile(r"""["']([\w./~-]+\.(?:csv|tsv|parquet|json|joblib|pkl|yaml|yml|txt))["']""")
YAML_PATH = re.compile(r"^\s*path:\s*(\S+)\s*$")


@dataclass
class FilterConfig:
    require_judge: bool = True
    max_steps: int = 12
    max_diff_factor: float = 4.0          # lignes changées <= max(min_diff_lines, facteur x lignes de l'injection)
    min_diff_lines: int = 10
    allow_compacted: bool = False


def changed_lines(diff: str) -> tuple[int, int, list[str]]:
    """(lignes ajoutées, lignes retirées, fichiers touchés) d'un diff unifié."""
    added = removed = 0
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            files.append(line[4:].strip().removeprefix("b/"))
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed, [f for f in files if f != "/dev/null"]


def invented_paths(diff: str, root: Path) -> list[str]:
    """Chemins de fichiers INTRODUITS par le diff qui n'existent pas dans le projet réparé (ou absolus hors projet)."""
    bad = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:]
        cands = PATH_LITERAL.findall(body) + ([m.group(1)] if (m := YAML_PATH.match(body)) else [])
        for c in cands:
            c = c.strip("'\"")
            if c.startswith(("http://", "https://")):
                continue
            if c.startswith(("/", "~")) or not (root / c).exists():
                bad.append(c)
    return sorted(set(bad))


def repair_filters(run: dict, cfg: FilterConfig) -> list[str]:
    r, why = run["repair"], []
    base = set(run.get("baseline_issue_ids", []))                     # problèmes déjà présents sur le projet SAIN (ex. sandbox non isolé)
    unresolved = [u for u in r.get("unresolved_ids", []) if u not in base]
    if r["status"] not in ("REPAIRED", "PARTIAL") or r.get("failure"):
        return [f"repair_status:{r['status']}"]
    if r["status"] == "PARTIAL" and unresolved:
        why.append("not_all_fixed:" + ",".join(unresolved))
    if any(t["status"] in ("FAIL", "ERROR") for t in r.get("tests", [])):
        why.append("tests_failed")
    if r.get("scope_flags"):                                          # `requires_human_approval` n'est pas testé seul : ses causes (flags, métriques, problèmes
        why.append("scope_flags")                                     # non résolus) sont vérifiées une par une ; un problème du projet SAIN le rendrait toujours vrai
    if r.get("behavior_changed"):
        why.append("behavior_changed")
    gold = run["gold"]
    off = sorted(set(r.get("files_changed", [])) - set(gold["files"]))
    if off:
        why.append("off_target_files:" + ",".join(off))
    if r.get("added", 0) + r.get("removed", 0) > max(cfg.min_diff_lines, cfg.max_diff_factor * gold["changed_lines"]):
        why.append("diff_too_large")
    if r.get("invented_paths"):
        why.append("invented_path:" + ",".join(r["invented_paths"]))
    if r.get("steps", 0) > cfg.max_steps:
        why.append("too_many_steps")
    if not r.get("files_changed") and not r.get("removed_artifacts"):
        why.append("no_change")
    return why


def revalidation_filters(run: dict) -> list[str]:
    v = run.get("revalidation")
    if v is None:
        return ["not_revalidated"]
    why = []
    if v["status"] not in OK_STATUSES and v["status"] != run.get("baseline_status"):
        why.append(f"revalidation:{v['status']}")
    remaining = sorted(set(v.get("issue_checks", [])) - set(run.get("baseline_checks", [])))
    if remaining:
        why.append("issues_remaining:" + ",".join(remaining))
    return why


def judge_filters(run: dict, cfg: FilterConfig) -> list[str]:
    j = run.get("judge")
    if j is None:
        return ["no_judge"] if cfg.require_judge else []
    why = []
    if j["verdict"] != "ACCEPT":
        why.append(f"judge:{j['verdict']}")
    if j.get("independent_model") is False:
        why.append("judge_not_independent")
    if j.get("failure"):
        why.append("judge_failed")
    return why


def trajectory_filters(run: dict, cfg: FilterConfig) -> list[str]:
    t = run.get("trajectory")
    if not t:
        return ["no_trajectory"]
    return ["compacted_trajectory"] if t.get("compacted") and not cfg.allow_compacted else []


def evaluate(run: dict, cfg: FilterConfig | None = None) -> list[str]:
    """Toutes les raisons d'écarter l'exécution (vide = exemple retenu)."""
    cfg = cfg or FilterConfig()
    return repair_filters(run, cfg) + revalidation_filters(run) + judge_filters(run, cfg) + trajectory_filters(run, cfg)
