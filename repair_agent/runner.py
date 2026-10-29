"""Lanceur de tests du Repair : syntaxe, analyse statique, dépendances, import, tests unitaires du projet, comparaison de métriques.

Exécuté PAR L'INFRASTRUCTURE après la boucle (on ne croit pas le LLM sur parole), et à la demande du LLM (`run_checks`).
Tout ce qui s'exécute passe par le sandbox, sur une copie du projet.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from agentic_core.facts import python_sources, static_facts
from agentic_core.sandbox_exec import SandboxRunner, metrics_delta, num
from agentic_core.schemas import MetricsComparison, TestResult
from validator.check_structure import _is_pinned, parse_requirements
from validator.fs import walk_project

EXCERPT = 400


def _t(name, status, command="", code=None, out=""):
    return TestResult(name=name, status=status, command=command, exit_code=code, output_excerpt=(out or "")[:EXCERPT])


def local_modules(root: Path) -> list[str]:
    return sorted(p.stem for p in root.glob("*.py") if not p.name.startswith(("test_", "_")) and not p.name.endswith("_test.py") and p.name != "conftest.py")


def run_checks(root: Path, runner: SandboxRunner | None, names=("syntax", "static", "dependencies", "import", "unit_tests")) -> list[TestResult]:
    root = Path(root)
    files, _ = walk_project(root)
    out: list[TestResult] = []
    if "syntax" in names:
        bad = []
        for rel, text in python_sources(root, files).items():
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as e:
                bad.append(f"{rel}:{e.lineno} {e.msg}")
        out.append(_t("syntax", "FAIL" if bad else "PASS", "ast.parse sur tous les .py et cellules de notebooks", 1 if bad else 0, "; ".join(bad) or "tous les fichiers se parsent"))
    if "static" in names:
        f = static_facts(root, files)
        keys = [k for k in ("dangerous_calls", "network_imports", "network_urls", "secrets", "sensitive_files", "interactive_input", "absolute_paths") if f.get(k)]
        out.append(_t("static", "FAIL" if keys else "PASS", "analyse statique (AST, sans exécution)", 1 if keys else 0,
                      "; ".join(f"{k}: {f[k][0]}" for k in keys) or "aucun fait dangereux, interactif ou chemin absolu"))
    if "dependencies" in names:
        req = root / "requirements.txt"
        if not req.is_file():
            out.append(_t("dependencies", "SKIPPED", "lecture de requirements.txt", None, "pas de requirements.txt"))
        else:
            loose = [raw for _, raw, r, _e in parse_requirements(req.read_text(encoding="utf-8", errors="replace")) if r is None or not _is_pinned(r)]
            out.append(_t("dependencies", "FAIL" if loose else "PASS", "requirements.txt : tout épinglé avec ==", 1 if loose else 0, "non épinglé : " + ", ".join(loose[:6]) if loose else "toutes les dépendances sont épinglées"))
    needs_exec = [n for n in ("import", "unit_tests") if n in names]
    if needs_exec and runner is None:
        out += [_t(n, "SKIPPED", "", None, "aucun sandbox d'exécution configuré") for n in needs_exec]
    elif needs_exec:
        if "import" in names:
            mods = local_modules(root)
            r = runner.run("import_check", root=root, modules=mods) if mods else {"skipped": "aucun module"}
            out.append(_res("import", r, f"import des modules {mods}"))
        if "unit_tests" in names:
            out.append(_res("unit_tests", runner.run("pytest", root=root), "python -m pytest -x -q"))
    return out


def _res(name, r: dict, cmd: str) -> TestResult:
    if r.get("not_run"):
        return _t(name, "SKIPPED", cmd, None, r.get("error", ""))
    if r.get("skipped"):
        return _t(name, "SKIPPED", cmd, None, r["skipped"])
    ok = r.get("ok")
    return _t(name, "PASS" if ok else "FAIL", cmd, r.get("returncode"), (r.get("stderr_tail") or r.get("stdout_tail") or ("ok" if ok else "")))


def compare_metrics(source: Path, repaired: Path, runner: SandboxRunner | None, spec, profile: str | None, tolerance: float) -> MetricsComparison:
    """Réparation qui change le comportement ? Métriques du harnais avant (original) / après (copie corrigée)."""
    if runner is None or not profile or spec is None or not (spec.data_path and spec.target and spec.primary_metric):
        return MetricsComparison(status="NOT_AVAILABLE", reason="spécification du projet (données, cible, métrique) ou sandbox indisponible", tolerance=tolerance)
    p = dict(profile=profile, data_path=spec.data_path, target=spec.target, primary_metric=spec.primary_metric, check_determinism=False)
    after = runner.run("harness", root=repaired, **p)
    if after.get("not_run") or not after.get("executed_ok"):
        return MetricsComparison(status="NOT_AVAILABLE", reason="le projet corrigé ne s'exécute pas dans le harnais : " + str(after.get("error") or after.get("harness_error") or after.get("_summary"))[:200], tolerance=tolerance)
    a = num(after.get("metrics"))
    before = runner.run("harness", root=source, **p)
    if before.get("not_run") or not before.get("executed_ok"):
        return MetricsComparison(status="AFTER_ONLY", reason="l'original ne s'exécute pas (attendu s'il était cassé) : pas de comparaison", after=a, tolerance=tolerance)
    b = num(before.get("metrics"))
    d = metrics_delta(b, a)
    return MetricsComparison(status="COMPARED", before=b, after=a, delta=d, tolerance=tolerance, behavior_changed=any(abs(v) > tolerance for v in d.values()))
