"""Validation statique et structurelle de l'api.py generee (AVANT tout test runtime).

Controles, dans l'ordre (on s'arrete au premier echec bloquant) :
  1. api.py existe
  2. syntaxe valide (ast)
  3. lint ruff (erreurs reelles : E9, F)         -> ignore proprement si ruff est absent
  4. import de l'app dans le dossier d'execution  (detecte imports / classes manquants)
  5. routes /health (GET) et /predict (POST) presentes
  6. contrat OpenAPI == input_schema / output_schema du rapport
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .errors import classify, tail_relevant
from .schemas import AnalysisReport

_PROBE = r"""
import json
import api
routes = [{"path": r.path, "methods": sorted(getattr(r, "methods", None) or [])} for r in api.app.routes]
print("__PROBE__" + json.dumps({"routes": routes, "openapi": api.app.openapi()}))
"""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False
    category: Optional[str] = None  # categorie d'erreur (errors.py) si echec


@dataclass
class ValidationResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        """Texte compact a renvoyer au LLM pour la correction."""
        if self.ok:
            return "Validation OK"
        return "\n".join(f"[{c.category or 'API_CONTRACT'}] {c.name}: {c.detail}" for c in self.failures)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def _resolve(openapi: dict, schema: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    node = openapi
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def _check_contract(report: AnalysisReport, openapi: dict) -> list[Check]:
    out: list[Check] = []
    post = openapi.get("paths", {}).get("/predict", {}).get("post")
    if not post:
        return [Check("contract", False, "POST /predict absent de l'OpenAPI", category="API_ROUTE_ERROR")]

    # Entree
    try:
        body = post["requestBody"]["content"]["application/json"]["schema"]
        req = _resolve(openapi, body)
        props, required = set(req.get("properties", {})), set(req.get("required", []))
    except KeyError:
        return [Check("contract_input", False, "Schema de requete introuvable", category="API_SCHEMA_MISMATCH")]
    exp = {f.name for f in report.io.input_schema}
    exp_req = {f.name for f in report.io.input_schema if f.required}
    problems = []
    if exp - props:
        problems.append(f"champs manquants: {sorted(exp - props)}")
    if props - exp:
        problems.append(f"champs en trop: {sorted(props - exp)}")
    if exp_req - required:
        problems.append(f"devraient etre obligatoires: {sorted(exp_req - required)}")
    out.append(Check("contract_input", not problems, "; ".join(problems), category=None if not problems else "API_SCHEMA_MISMATCH"))

    # Sortie
    try:
        resp = _resolve(openapi, post["responses"]["200"]["content"]["application/json"]["schema"])
        out_props = set(resp.get("properties", {}))
    except KeyError:
        out_props = set()
    exp_out = {f.name for f in report.io.output_schema if not f.optional}
    missing = exp_out - out_props
    out.append(Check("contract_output", not missing, f"champs de sortie manquants: {sorted(missing)}" if missing else "",
                     category="API_CONTRACT" if missing else None))
    return out


def validate_api(report: AnalysisReport, run_dir: Path, python_exe: Optional[str] = None, timeout: int = 120) -> ValidationResult:
    """run_dir = dossier contenant api.py + code du projet (voir api_tests.prepare_run_dir)."""
    py = python_exe or sys.executable
    run_dir = Path(run_dir)
    res = ValidationResult()
    api_file = run_dir / "api.py"

    # 1. existence
    if not api_file.is_file():
        res.checks.append(Check("file_exists", False, f"{api_file} introuvable", category="API_SYNTAX"))
        return res
    res.checks.append(Check("file_exists", True))

    # 2. syntaxe
    try:
        ast.parse(api_file.read_text(encoding="utf-8"))
        res.checks.append(Check("syntax", True))
    except SyntaxError as e:
        res.checks.append(Check("syntax", False, f"ligne {e.lineno}: {e.msg}", category="API_SYNTAX"))
        return res

    # 3. lint (non bloquant si ruff absent)
    lint = subprocess.run([py, "-m", "ruff", "check", "--select", "E9,F", "--no-cache", "--output-format", "concise", "api.py"],
                          cwd=run_dir, capture_output=True, text=True, timeout=timeout, env=config.clean_env())
    if "No module named ruff" in lint.stderr:
        res.checks.append(Check("lint", True, "ruff non installe : controle ignore", skipped=True))
    elif lint.returncode != 0:
        res.checks.append(Check("lint", False, lint.stdout.strip()[-800:], category="API_IMPORT"))
        return res
    else:
        res.checks.append(Check("lint", True))

    # 4. import + extraction OpenAPI
    proc = subprocess.run([py, "-c", _PROBE], cwd=run_dir, capture_output=True, text=True, timeout=timeout, env=config.clean_env())
    marker = next((l for l in proc.stdout.splitlines() if l.startswith("__PROBE__")), None)
    if proc.returncode != 0 or marker is None:
        logs = proc.stderr or proc.stdout
        cat, _ = classify(logs)
        if cat in ("UNKNOWN", "DEP_MISSING_PYTHON"):  # module manquant : import fautif ou dependance, le LLM tranchera
            cat = "API_IMPORT"
        res.checks.append(Check("import_app", False, tail_relevant(logs, 25), category=cat))
        return res
    res.checks.append(Check("import_app", True))
    probe = json.loads(marker[len("__PROBE__"):])

    # 5. routes
    have = {(r["path"], m) for r in probe["routes"] for m in r["methods"]}
    missing = [x for x in [("/health", "GET"), ("/predict", "POST")] if x not in have]
    res.checks.append(Check("routes", not missing, f"routes absentes: {missing}" if missing else "",
                            category="API_ROUTE_ERROR" if missing else None))

    # 6. contrat
    res.checks.extend(_check_contract(report, probe["openapi"]))
    return res
