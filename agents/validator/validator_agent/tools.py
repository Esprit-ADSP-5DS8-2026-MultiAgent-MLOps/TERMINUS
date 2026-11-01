"""Outils du Validator : ils renvoient des FAITS ; le LLM interprète, le moteur de politique décide du statut."""
from __future__ import annotations

import tomllib
from typing import get_args
from pathlib import Path

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype
from packaging.requirements import Requirement

from agentic_core.facts import static_facts, unsafe_to_execute
from agentic_core.sandbox_exec import COMMANDS, SandboxRunner
from agentic_core.schemas import Category, CheckStatus, Nature, Severity
from agentic_core.toolkit import BaseToolBox, ToolError, fn_spec, j
from validator.check_structure import _is_pinned, parse_requirements
from validator.entrypoints import ROLES_BY_HARNESS
from validator_agent.proposals import PROPOSALS_JSON_SCHEMA

DEP_FILES = ("pyproject.toml", "environment.yml", "environment.yaml", "Pipfile", "setup.py", "setup.cfg")


def _req_fact(raw: str):
    try:
        r = Requirement(raw)
        return {"name": r.name, "spec": str(r.specifier) or None, "pinned": _is_pinned(r)}
    except Exception:
        return {"raw": raw[:80], "unparsable": True}


class ValidatorToolBox(BaseToolBox):
    def __init__(self, root, contract, runner: SandboxRunner | None = None, *, sandbox="auto", env="venv", cache_dir=None, timeout=300, max_data_mb=50):
        super().__init__(root)
        self.contract = contract
        self.runner = runner or SandboxRunner(root, sandbox=sandbox, env=env, cache_dir=cache_dir, timeout=timeout)
        self.max_data_mb = max_data_mb
        self.executed_ok: bool | None = None          # None = jamais tenté ; True/False = résultat du harnais
        self.harness_result: dict | None = None
        self.sandbox_info: dict | None = None
        self.limits_seen: list[str] = []
        self._facts: dict | None = None
        self._eps = None
        self.hints: dict = {}                      # hypothèses du LLM CONFIRMÉES par Python (spec, entrypoints, métrique)
        self.llm_can_execute = True                # passe à False pendant la boucle LLM du mode déterministe : Python est le seul à exécuter
        self.tools.update(analyze_code=self.analyze_code, read_dependency_config=self.read_dependency_config,
                          inspect_data=self.inspect_data, run_in_sandbox=self.run_in_sandbox)

    def specs(self):
        return [s for s in self._all_specs() if self.llm_can_execute or s.get("function", s).get("name") != "run_in_sandbox"]

    def _all_specs(self):
        return self.common_specs() + [
            fn_spec("analyze_code", "Faits statiques (AST, sans exécuter) sur tous les .py ET les cellules de code des notebooks : erreurs de syntaxe, imports "
                    "tiers, appels dangereux, imports/URL réseau, input(), chemins absolus, secrets (masqués), graines manquantes, fonctions et paramètres, "
                    "artefacts binaires. Ce sont des FAITS : c'est à toi de les interpréter.", {}),
            fn_spec("read_dependency_config", "Faits sur les dépendances déclarées : requirements*.txt, pyproject.toml, environment.yml, Pipfile "
                    "(nom, spécificateur, épinglé ou non). Sans argument : tous les fichiers détectés.", {"path": {"type": "string"}}),
            fn_spec("inspect_data", "Profil d'un fichier de données (CSV/TSV, JSON lignes, parquet si possible) : lignes, colonnes, types, nuls, balance de la cible. "
                    "Jamais de lignes brutes. Signale ses LIMITES (format non lu) au lieu de conclure.",
                    {"path": {"type": "string"}, "target": {"type": "string", "description": "colonne cible, si connue"}}, ["path"]),
            fn_spec("run_in_sandbox", "Exécute le projet dans le sandbox, en commandes autorisées uniquement. "
                    "harness : exécute le contrat du profil (main.py + train + predict selon le profil) ; import_check : importe des modules locaux ; "
                    "entrypoint : lance un script .py du projet. REFUSÉ si analyze_code montre des faits dangereux (appels, réseau, secrets).",
                    {"command": {"type": "string", "enum": list(COMMANDS[:3])},
                     "data_path": {"type": "string"}, "target": {"type": "string"}, "primary_metric": {"type": "string"},
                     "modules": {"type": "array", "items": {"type": "string"}}, "script": {"type": "string"}}, ["command"]),
            fn_spec("submit_report", "Dépose le rapport FINAL (une seule fois). Le statut d'admission n'en fait pas partie : il est calculé par le moteur de politique.", {
                "project_type": {"type": "string", "description": "ex. tabular_ml, nlp, vision, time_series, notebook_only, ou « unknown » si tu ne peux pas le prouver"},
                "frameworks": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["name"]}},
                "checks": {"type": "array", "description": "UN élément par contrôle du catalogue", "items": {"type": "object", "properties": {
                    "check_id": {"type": "string"}, "status": {"type": "string", "enum": [s.value for s in CheckStatus]},
                    "severity": {"type": "string", "enum": [s.value for s in Severity]}, "file": {"type": "string"}, "line": {"type": "integer"},
                    "evidence": {"type": "string", "description": "extrait EXACT d'une sortie d'outil (obligatoire pour FAIL/WARNING/NOT_APPLICABLE)"},
                    "na_reason": {"type": "string", "description": "id de la condition NOT_APPLICABLE du contrat"}, "note": {"type": "string"}},
                    "required": ["check_id", "status"]}},
                "issues": {"type": "array", "items": {"type": "object", "properties": {
                    "check_id": {"type": "string"},
                    "category": {"type": "string", "enum": list(get_args(Category)),
                                 "description": "catégorie du problème, STRICTEMENT l'une des valeurs de l'énumération. "
                                                "Ce n'est PAS une nature : 'configuration' / 'mechanical' / 'logic' / 'methodological' vont dans `nature`."},
                    "severity": {"type": "string", "enum": [s.value for s in Severity]},
                    "file": {"type": "string"}, "line": {"type": "integer"}, "description": {"type": "string"}, "suggested_fix": {"type": "string"},
                    "nature": {"type": "string", "enum": [s.value for s in Nature]}, "evidence": {"type": "string"}},
                    "required": ["check_id", "description", "evidence"]}},
                "limits": {"type": "array", "items": {"type": "string"}, "description": "ce que l'analyse n'a PAS pu couvrir"},
                "project_spec": {"type": "object", "properties": {"task": {"type": "string", "enum": ["classification", "regression"]}, "target": {"type": "string"},
                                 "primary_metric": {"type": "string"}, "min_threshold": {"type": "number"}, "data_path": {"type": "string"}}},
                "proposals": PROPOSALS_JSON_SCHEMA,
                "summary": {"type": "string"}}, ["project_type", "checks"]),
        ]

    # ------------------------------------------------------------------ outils
    def entrypoint_resolution(self):
        """Entrypoints résolus par Python (manifeste -> AST -> convention). Jamais décidés par le LLM."""
        if self._eps is None:
            from validator.entrypoints import read_manifest_body, resolve_entrypoints
            self._eps = resolve_entrypoints(self.root, sorted(self.known_files), read_manifest_body(self.root), self.contract.harness)
            for role, entry in (self.hints.get("entrypoints") or {}).items():
                self._eps.entries[role] = entry
                self._eps.problems = [p for p in self._eps.problems if p.role != role]
        return self._eps

    def facts(self) -> dict:
        if self._facts is None:
            self._facts = static_facts(self.root)
            self._facts["entrypoints"] = self.entrypoint_resolution().summary()
        return self._facts

    def analyze_code(self) -> str:
        return j({k: v for k, v in self.facts().items() if not k.startswith("_")})     # « _ » : données d'infrastructure, pas pour le LLM

    def read_dependency_config(self, path: str | None = None) -> str:
        names = [path] if path else sorted(f for f in self.known_files if "/" not in f and (
            f.startswith("requirements") and f.endswith(".txt") or f in DEP_FILES))
        out = []
        for n in names:
            p = self.resolve(n)
            if not p.is_file():
                out.append({"file": n, "error": "introuvable"})
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.name.startswith("requirements") and p.suffix == ".txt":
                deps = []
                for ln, raw, req, err in parse_requirements(text):
                    deps.append({"line": ln, **({"raw": raw[:80], "unparsable": True, "why": err[:60]} if req is None else
                                 {"name": req.name, "spec": str(req.specifier) or None, "pinned": _is_pinned(req)})})
                out.append({"file": n, "kind": "requirements", "dependencies": deps})
            elif p.name == "pyproject.toml":
                try:
                    d = tomllib.loads(text)
                    proj = d.get("project", {})
                    out.append({"file": n, "kind": "pyproject", "requires_python": proj.get("requires-python"),
                                "dependencies": [_req_fact(x) for x in proj.get("dependencies", [])],
                                "optional_groups": sorted((proj.get("optional-dependencies") or {})),
                                "build_backend": (d.get("build-system") or {}).get("build-backend"),
                                "poetry_dependencies": sorted((d.get("tool", {}).get("poetry", {}).get("dependencies") or {}))[:30] or None})
                except Exception as e:
                    out.append({"file": n, "kind": "pyproject", "error": f"TOML invalide : {type(e).__name__}"})
            elif p.suffix in (".yml", ".yaml"):
                try:
                    d = yaml.safe_load(text) or {}
                    conda, pip = [], []
                    for x in d.get("dependencies", []) or []:
                        if isinstance(x, dict):
                            pip += [_req_fact(y) for y in x.get("pip", [])]
                        else:
                            conda.append(str(x))
                    out.append({"file": n, "kind": "conda_environment", "conda_dependencies": conda[:40], "pip_dependencies": pip,
                                "channels": d.get("channels")})
                except Exception as e:
                    out.append({"file": n, "kind": "conda_environment", "error": f"YAML invalide : {type(e).__name__}"})
            else:
                out.append({"file": n, "kind": "other", "note": "format non analysé en détail : première lignes uniquement", "head": text[:200]})
        if not out:
            self.limits_seen.append("aucun fichier de dépendances détecté")
            return j({"dependency_files": [], "note": "aucun requirements*.txt / pyproject.toml / environment.yml à la racine"})
        return j({"dependency_files": out})

    def inspect_data(self, path: str, target: str | None = None) -> str:
        p = self.resolve(path)
        if not p.is_file():
            return j({"error": f"introuvable : {path}"})
        mb = p.stat().st_size / 1e6
        if mb > self.max_data_mb:
            return j({"path": path, "size_mb": round(mb, 1), "limit": f"fichier > {self.max_data_mb} Mo : non lu"})
        suf = p.suffix.lower()
        try:
            if suf in (".csv", ".tsv"):
                df = pd.read_csv(p, sep="\t" if suf == ".tsv" else ",", encoding="utf-8", low_memory=False)
            elif suf in (".json", ".jsonl"):
                df = pd.read_json(p, lines=suf == ".jsonl")
            elif suf == ".parquet":
                df = pd.read_parquet(p)
            else:
                lim = f"format {suf or 'sans extension'} non pris en charge par l'inspecteur"
                self.limits_seen.append(f"{path} : {lim}")
                return j({"path": path, "readable": False, "limit": lim})
        except UnicodeDecodeError:
            return j({"path": path, "readable": False, "problem": "encodage non UTF-8"})
        except ImportError as e:
            lim = f"lecteur indisponible ({e.name})"
            self.limits_seen.append(f"{path} : {lim}")
            return j({"path": path, "readable": False, "limit": lim})
        except Exception as e:
            return j({"path": path, "readable": False, "problem": f"{type(e).__name__}: {str(e)[:120]}"})
        cols = [{"name": str(c), "numeric": bool(is_numeric_dtype(df[c])), "nulls": int(df[c].isna().sum()), "unique": int(df[c].nunique())} for c in df.columns[:40]]
        out = {"path": path, "readable": True, "rows": len(df), "n_columns": len(df.columns), "columns": cols,
               "duplicate_rows": int(df.duplicated().sum())}
        if len(df.columns) > 40:
            out["note"] = "40 premières colonnes seulement"
        if target:
            if target not in df.columns:
                out["target"] = {"name": target, "present": False}
            else:
                s = df[target]
                t = {"name": target, "present": True, "nulls": int(s.isna().sum()), "unique": int(s.nunique()), "numeric": bool(is_numeric_dtype(s))}
                if s.nunique() <= 20:
                    t["class_counts"] = {str(k): int(v) for k, v in s.value_counts().items()}
                out["target"] = t
        return j(out)

    def run_in_sandbox(self, command: str, **p) -> str:
        if command not in COMMANDS[:3]:
            raise ToolError(f"commande non autorisée : {command}")
        if not self.llm_can_execute:                      # basculé par l'agent PENDANT la boucle LLM seulement ; le LLM ne peut pas le rétablir
            raise ToolError("outil réservé à Python dans ce mode : propose spec/entrypoints dans `proposals`, Python les vérifiera en exécutant")
        bad = unsafe_to_execute(self.facts())
        if bad:
            return j({"refused": True, "reason": "exécution refusée : faits statiques dangereux", "facts": bad,
                      "instruction": "ne lance PAS le projet ; marque EXEC.RUN NOT_CHECKED avec cette raison, et rapporte les contrôles SEC.* concernés."})
        if command == "harness":
            if not self.contract.harness:
                raise ToolError("ce profil ne prévoit pas d'exécution")
            if not all(isinstance(p.get(k), str) and p.get(k) for k in ("data_path", "target")):
                raise ToolError("harness exige data_path et target (déduis-les du manifeste ou du code, ou déclare EXEC.RUN NOT_CHECKED)")
            pm = p.get("primary_metric") if isinstance(p.get("primary_metric"), str) and p.get("primary_metric") else None   # None : lue dans le dict renvoyé par train()
            extra = {}
            if self.hints.get("entrypoints"):             # entrypoints confirmés par Python : le runner ne les retrouverait pas seul
                extra["entrypoints"] = self.entrypoint_resolution().harness_map(ROLES_BY_HARNESS.get(self.contract.harness, ("train",)))
            res = self.runner.run("harness", profile=self.contract.harness, data_path=p["data_path"], target=p["target"], primary_metric=pm, **extra)
        elif command == "import_check":
            res = self.runner.run("import_check", modules=p.get("modules"))
        else:
            res = self.runner.run("entrypoint", script=p.get("script"))
        if self.runner.info:
            self.sandbox_info = self.runner.info
        if res.get("not_run"):
            self.limits_seen.append(f"exécution impossible : {res.get('error')}")
        elif command == "harness":
            self.harness_result = res
            self.executed_ok = bool(res.get("executed_ok"))
        return j(res)
