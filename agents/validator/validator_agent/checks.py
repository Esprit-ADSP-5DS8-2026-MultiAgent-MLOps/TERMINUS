"""Contrôles déterministes du Validator : Python collecte les preuves et statue ; le LLM n'intervient pas ici.

Les identifiants, le caractère obligatoire et `on_fail` restent ceux du contrat YAML. Ce module n'invente
pas de décision d'admission : il produit des `Check` / `Issue` pour `policy.engine.evaluate`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

from agentic_core.facts import unsafe_to_execute
from agentic_core.introspection import (audit_findings, build_request, describe_event, dist_for_import, import_status, module_dists,
                                        seed_suspects, unknown_estimator_frameworks)
from agentic_core.schemas import Category, Check, CheckStatus, Issue, Nature, ProjectSpec, Severity
from agentic_core.toolkit import j
from policy.model import AdmissionContract, ContractCheck
from validator.contract import ContractConfig, norm_dist
from validator.check_structure import _is_pinned, parse_requirements
from validator.entrypoints import DEFAULT_FILE, ROLES_BY_HARNESS
from validator_agent.spec_infer import deduce_spec

# Limites «nelles de STRUCT.LIMITS (« taille raisonnable ») : mêmes plafonds que la baseline `validator/`.
_LIMITS = ContractConfig()
ALLOWED_CATEGORIES = set(get_args(Category))
METRIC_HINTS = ("accuracy_score", "roc_auc_score", "f1_score", "precision_score", "recall_score",
                "mean_squared_error", "mean_absolute_error", "r2_score", "log_loss", ".score(")
METRIC_KEYS = frozenset({"accuracy", "roc_auc", "f1", "precision", "recall", "rmse", "mae", "r2", "log_loss"})
HARDCODED_METRIC = re.compile(
    r"""return\s*\{[^}]{0,200}['"](?:accuracy|roc_auc|f1|precision|recall|rmse|mae|r2)['"]\s*:\s*\d""",
    re.I | re.S,
)
BLOCKING_SEC = ("SEC.DANGEROUS_CALLS", "SEC.SECRETS", "SEC.NETWORK")
LLM_REVIEW_IDS = frozenset({"ML.METRICS_GENUINE"})


def _cat(spec: ContractCheck | None) -> str:
    c = (spec.category if spec else "other") or "other"
    return c if c in ALLOWED_CATEGORIES else "other"


def _nature(spec: ContractCheck | None) -> Nature:
    try:
        return Nature((spec.nature if spec else "mechanical") or "mechanical")
    except ValueError:
        return Nature.MECHANICAL


def _sev(spec: ContractCheck | None, default=Severity.MAJOR) -> Severity:
    try:
        return Severity((spec.severity if spec else default.value) or default.value)
    except ValueError:
        return default


@dataclass
class DeterministicOutcome:
    checks: list[Check] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    spec: ProjectSpec | None = None
    limits: list[str] = field(default_factory=list)
    security_blocked: bool = False
    executed_ok: bool | None = None
    entrypoints: dict | None = None


def _mk(spec: ContractCheck, status: CheckStatus, *, evidence=None, note=None, file=None, line=None,
        source="python", origin="candidate") -> Check:
    return Check(check_id=spec.id, name=spec.title, level=spec.level, status=status, severity=_sev(spec),
                 evidence=evidence, note=note, file=file, line=line, source=source, origin=origin)


def _issue(spec: ContractCheck, description: str, *, evidence=None, file=None, line=None, suggested=None) -> Issue:
    return Issue(issue_id="", check_id=spec.id, category=_cat(spec), severity=_sev(spec), file=file, line=line,
                 description=description, suggested_fix=suggested, nature=_nature(spec), evidence=evidence)


def load_manifest(root) -> dict:
    for name in ("mlops.yaml", "mlops.yml"):
        p = root / name
        if p.is_file():
            try:
                import yaml
                return {"file": name, "data": yaml.safe_load(p.read_text(encoding="utf-8")) or {}}
            except Exception as e:
                return {"file": name, "error": f"{type(e).__name__}: {e}"}
    return {}


def infer_project_spec(tb, manifest: dict) -> ProjectSpec | None:
    yaml_body = manifest.get("data") if isinstance(manifest.get("data"), dict) else {}
    dsec = yaml_body.get("data") if isinstance(yaml_body.get("data"), dict) else {}
    esec = yaml_body.get("evaluation") if isinstance(yaml_body.get("evaluation"), dict) else {}
    psec = yaml_body.get("project") if isinstance(yaml_body.get("project"), dict) else {}
    path = dsec.get("path") if isinstance(dsec.get("path"), str) else None
    target = dsec.get("target") if isinstance(dsec.get("target"), str) else None
    metric = esec.get("primary_metric") if isinstance(esec.get("primary_metric"), str) else None
    task = psec.get("task") if psec.get("task") in ("classification", "regression") else None
    thr = esec.get("min_threshold") if isinstance(esec.get("min_threshold"), (int, float)) else None
    if not any((path, target, metric, task)):
        return None
    return ProjectSpec(task=task, target=target, primary_metric=metric, min_threshold=thr, data_path=path)


def required_files_of(contract: AdmissionContract) -> tuple[str, ...]:
    if contract.required_files:
        return tuple(contract.required_files)
    if contract.harness == "train_only":
        return ("train.py", "requirements.txt")
    if contract.harness == "sklearn_tabular":
        return ("main.py", "train.py", "predict.py", "requirements.txt")
    return ()


FIXED_ROLE_FILES = frozenset(DEFAULT_FILE.values()) | {"main.py"}


def required_files_for(contract: AdmissionContract, eps) -> tuple[list[str], list]:
    """Fichiers requis DÉDUITS des entrypoints résolus (src/, fit(), manifeste…). Le contrat ne garde que ce qui n'est pas un
    fichier de rôle (ex. requirements.txt). Sans résolution possible : noms par défaut du contrat. Retourne (fichiers, ambiguïtés)."""
    base = list(required_files_of(contract))
    roles = ROLES_BY_HARNESS.get(contract.harness or "")
    if eps is None or not roles:
        return base, []
    derived = eps.required_files(roles)
    return derived + [f for f in base if f not in FIXED_ROLE_FILES and f not in derived], eps.ambiguous(roles)


def _is_numeric_const(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_numeric_const(node.operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        return _is_numeric_const(node.left) and _is_numeric_const(node.right)
    return False


def _is_metric_key(name: str) -> bool:
    sl = name.lower()
    return sl in METRIC_KEYS or any(k in sl for k in METRIC_KEYS)


def inspect_train_metrics(text: str, fn_name: str = "train") -> tuple[CheckStatus, str | None, str]:
    """Statue ML.METRICS_GENUINE d'après ce que train() RENVOIE, pas d'après les seuls imports sklearn.metrics."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return CheckStatus.NOT_CHECKED, None, "train.py illisible (AST)"
    train_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if train_fn is None:
        return CheckStatus.NOT_CHECKED, None, f"pas de fonction {fn_name}()"
    assigned: dict[str, ast.AST] = {}
    for n in ast.walk(train_fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = n.value

    def resolve(node: ast.AST) -> ast.AST:
        return assigned[node.id] if isinstance(node, ast.Name) and node.id in assigned else node

    hardcoded: list[tuple[str, int]] = []
    computed = False
    for n in ast.walk(train_fn):
        if not isinstance(n, ast.Return) or n.value is None:
            continue
        items = resolve(n.value)
        nodes = items.elts if isinstance(items, (ast.Tuple, ast.List)) else [items]
        for item in nodes:
            item = resolve(item)
            if isinstance(item, ast.Dict):
                for k, v in zip(item.keys, item.values):
                    if not (isinstance(k, ast.Constant) and isinstance(k.value, str) and _is_metric_key(k.value)):
                        continue
                    v = resolve(v)
                    if _is_numeric_const(v):
                        hardcoded.append((k.value, n.lineno))
                    else:
                        computed = True
            elif isinstance(item, ast.DictComp):
                if _is_numeric_const(item.value):
                    hardcoded.append(("<comp>", n.lineno))
                else:
                    computed = True
            elif not isinstance(item, ast.Constant):
                computed = True
    if hardcoded:
        key, line = hardcoded[0]
        ev = f"train() ligne {line} : métrique {key} renvoyée en littéral numérique"
        return CheckStatus.FAIL, ev, ev
    if computed and any(h in text for h in METRIC_HINTS):
        return CheckStatus.PASS, None, "train() renvoie une expression, pas un littéral de métrique"
    if computed:
        return CheckStatus.NOT_CHECKED, None, "train() renvoie une expression sans appel de métrique visible"
    if HARDCODED_METRIC.search(text):
        ev = HARDCODED_METRIC.search(text).group(0)[:160]
        return CheckStatus.FAIL, ev, "retour de métriques littérales (regex)"
    return CheckStatus.NOT_CHECKED, None, "cas ambigu : à interpréter par le LLM"


def classify_exec_error(res: dict, *, env: str, declared: set[str]) -> str:
    """Origine d'un échec d'exécution : candidat, harnais, sandbox, ou validateur."""
    if res.get("not_run"):
        err = (res.get("error") or "").lower()
        if "sandbox indisponible" in err:
            return "sandbox"
        if "installation des dépendances" in err or "requirements.txt absent" in err:
            return "harness"
        return "harness"
    chunks = [str(res.get(k) or "") for k in ("harness_error", "stderr_tail", "_summary", "timeout", "error", "main")]
    steps = res.get("contract_steps")
    if isinstance(steps, dict):
        chunks.extend(str(v) for v in steps.values() if v)
    text = " ".join(chunks)
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", text) or re.search(r"ModuleNotFoundError: ([A-Za-z0-9_]+)", text)
    if m:
        mod = m.group(1).split(".")[0]
        dist = dist_for_import(mod)
        if env == "current" and dist in declared:
            return "validator"
        if dist not in declared:
            return "candidate"
        return "harness"
    if re.search(r"MemoryError|Cannot allocate memory|\bKilled\b|Errno 12", text):
        return "sandbox"                # limite de ressources du sandbox : pas une preuve que le projet est défectueux
    failed_steps = isinstance(steps, dict) and any(str(v).startswith("FAILED") for v in steps.values())
    main = res.get("main")
    main_failed = isinstance(main, dict) and main.get("returncode") not in (0, None)
    if res.get("harness_error") and not failed_steps and not main_failed:
        return "harness"                # le harnais a planté sans qu'aucune étape du projet n'ait échoué
    return "candidate"


def _deps(tb) -> tuple[set[str], list[dict]]:
    names, rows = set(), []
    req = tb.root / "requirements.txt"
    if not req.is_file():
        return names, rows
    for ln, raw_line, parsed, err in parse_requirements(req.read_text(encoding="utf-8", errors="replace")):
        if parsed is None:
            rows.append({"line": ln, "raw": raw_line, "ok": False, "err": err})
        else:
            names.add(norm_dist(parsed.name))
            rows.append({"line": ln, "raw": raw_line, "ok": True, "name": parsed.name, "pinned": _is_pinned(parsed), "spec": str(parsed.specifier)})
    return names, rows


def run_static_checks(tb, contract: AdmissionContract) -> DeterministicOutcome:
    """Contrôles statiques + spec de projet. N'exécute pas le candidat."""
    out = DeterministicOutcome()
    facts = tb.facts()
    tb.observe(j(facts))
    if not any(c["tool"] == "list_files" for c in tb.calls):
        tb.call("list_files", {})
    files = tb.known_files
    man = load_manifest(tb.root)
    out.spec = infer_project_spec(tb, man)
    declared, dep_rows = _deps(tb)
    eps = tb.entrypoint_resolution()
    out.entrypoints = eps.summary()
    out.spec = deduce_spec(tb, out.spec, eps)      # comble les trous du manifeste (ou son absence) avec des valeurs VÉRIFIÉES
    roles = ROLES_BY_HARNESS.get(contract.harness or "", ())

    def spec(cid):
        return contract.check(cid)

    def add(chk: Check, issue: Issue | None = None):
        if contract.check(chk.check_id) is None:
            return
        out.checks.append(chk)
        if issue and chk.status == CheckStatus.FAIL:
            ev = issue.evidence or chk.evidence
            if ev and not tb.is_grounded(ev):
                tb.observe(ev)
                issue.evidence = ev
            out.issues.append(issue)

    # --- STRUCT.REQUIRED_FILES : déduits des entrypoints résolus (manifeste -> AST -> convention)
    if (s := spec("STRUCT.REQUIRED_FILES")):
        req, amb = required_files_for(contract, eps)
        missing = [n for n in req if n not in files]
        invalid = [p for p in eps.problems if p.kind == "invalid" and p.role in roles]
        if not req:
            add(_mk(s, CheckStatus.NOT_CHECKED, note="contrat sans required_files", origin="validator"))
        elif invalid:
            ev = "; ".join(f"entrypoint invalide:{p.detail}" for p in invalid)
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=ev),
                _issue(s, f"Entrypoint déclaré invalide : {invalid[0].detail}", evidence=ev, suggested="Corriger `entrypoints` dans mlops.yaml (chemin relatif au projet)."))
        elif missing:
            ev = "; ".join(f"absent:{n}" for n in missing)
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=f"absents : {missing}"),
                _issue(s, f"Fichiers requis absents : {missing}", evidence=ev, suggested=f"Ajouter {missing[0]} (ou déclarer le bon chemin dans `entrypoints` de mlops.yaml)."))
        elif amb:
            ev = "; ".join(f"{p.role}: {', '.join(p.candidates[:4])}" for p in amb)
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator",
                    note="entrypoints ambigus : déclarez-les dans mlops.yaml (entrypoints.train / entrypoints.predict) ou revue humaine"))
        else:
            add(_mk(s, CheckStatus.PASS, note=f"présents : {req} · entrypoints {', '.join(f'{r}={e.file}:{e.function} ({e.source})' for r, e in eps.entries.items())}"))

    # --- STRUCT.LIMITS
    if (s := spec("STRUCT.LIMITS")):
        n = len(files)
        total = sum((tb.root / f).stat().st_size for f in files if (tb.root / f).is_file())
        if n > _LIMITS.max_files:
            ev = f"total_files={n}"
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=f">{_LIMITS.max_files} fichiers"),
                _issue(s, f"{n} fichiers (max {_LIMITS.max_files}).", evidence=ev))
        elif total > _LIMITS.max_project_mb * 1e6:
            ev = f"size_mb={round(total / 1e6, 2)}"
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev),
                _issue(s, f"Projet de {total / 1e6:.0f} Mo (max {_LIMITS.max_project_mb}).", evidence=ev))
        else:
            add(_mk(s, CheckStatus.PASS, note=f"{n} fichiers, {round(total / 1e6, 2)} Mo"))

    # --- STRUCT.MANIFEST
    if (s := spec("STRUCT.MANIFEST")):
        if man.get("error"):
            ev = man["error"][:180]
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=man.get("file")),
                _issue(s, f"Manifeste illisible : {ev}", evidence=ev, file=man.get("file")))
        elif not man and out.spec and out.spec.data_path and out.spec.target and {"data_path", "target"} <= set(out.spec.deduced):
            d = out.spec.deduced
            add(_mk(s, CheckStatus.WARNING, note="mlops.yaml absent : spec DÉDUITE puis vérifiée par Python · "
                    + " · ".join(f"{k}={getattr(out.spec, k)} ({d[k]})" for k in ("data_path", "target") if k in d),
                    origin="validator"))
        elif not man:
            ev = "absent:mlops.yaml"
            tb.observe(ev)
            why = (out.spec.deduced.get("target?") if out.spec else None) or "données ou cible non déductibles avec certitude"
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=f"manifeste absent et spec non déductible ({why}) → revue humaine"),
                _issue(s, "Aucun mlops.yaml/yml ; tâche/cible/métrique non confirmées.", evidence=ev, suggested="Ajouter un manifeste mlops.yaml."))
        elif out.spec and (out.spec.data_path or out.spec.target):
            filled = sorted(k for k in out.spec.deduced if not k.endswith("?"))
            if filled:
                add(_mk(s, CheckStatus.WARNING, file=man.get("file"), origin="validator",
                        note=f"manifeste incomplet : {filled} déduits et vérifiés par Python"))
            else:
                add(_mk(s, CheckStatus.PASS, file=man.get("file"), note="manifeste complet"))
        else:
            ev = man.get("file") or "mlops.yaml"
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=man.get("file"), note="champs data/evaluation incomplets"),
                _issue(s, "Manifeste présent mais cible, données ou métrique absentes.", evidence=ev, file=man.get("file")))

    # --- CODE.SYNTAX
    if (s := spec("CODE.SYNTAX")):
        errs = facts.get("syntax_errors") or []
        if errs:
            ev = j(errs[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=errs[0].get("file"), line=errs[0].get("line")),
                _issue(s, f"Erreur de syntaxe : {errs[0].get('msg')}", evidence=ev, file=errs[0].get("file"), line=errs[0].get("line")))
        elif facts.get("python_files_analyzed"):
            add(_mk(s, CheckStatus.PASS, note=f"{len(facts['python_files_analyzed'])} fichiers analysés"))
        else:
            add(_mk(s, CheckStatus.NOT_CHECKED, note="aucun .py analysé", origin="validator"))

    # --- ENV.DEPENDENCIES_PINNED
    if (s := spec("ENV.DEPENDENCIES_PINNED")):
        if not (tb.root / "requirements.txt").is_file():
            ev = "absent:requirements.txt"
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file="requirements.txt"),
                _issue(s, "requirements.txt absent : dépendances non épinglées.", evidence=ev, file="requirements.txt"))
        else:
            bad = [r for r in dep_rows if r["ok"] and not r["pinned"]]
            unparsable = [r for r in dep_rows if not r["ok"] and r.get("err") != "option"]
            if bad or unparsable:
                hit = (bad or unparsable)[0]
                ev = hit["raw"][:120]
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file="requirements.txt", line=hit["line"]),
                    _issue(s, f"Dépendance non épinglée ou illisible : {ev}", evidence=ev, file="requirements.txt",
                           line=hit["line"], suggested=f"{hit.get('name', 'paquet')}==<version testée>"))
            else:
                add(_mk(s, CheckStatus.PASS, file="requirements.txt"))

    # --- ENV.DEPENDENCIES_DECLARED (recommandé) : estimation par métadonnées ; l'introspection du venv tranche après l'exécution
    if (s := spec("ENV.DEPENDENCIES_DECLARED")):
        mods = facts.get("_third_party_modules") or {}
        unsure = sorted(m for m in mods if dist_for_import(m) not in declared)
        if unsure:
            ev = ", ".join(unsure[:8])
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator",
                    note=f"import → paquet à confirmer dans le venv du projet (introspection) : {unsure[:8]}"))
        elif mods:
            add(_mk(s, CheckStatus.PASS, note=f"{len(mods)} imports tiers déclarés (estimation par métadonnées des paquets)"))
        else:
            add(_mk(s, CheckStatus.NOT_CHECKED, note="pas d'import tiers observé", origin="validator"))

    # --- FRAMEWORK.SCOPE
    if (s := spec("FRAMEWORK.SCOPE")):
        out_scope = {norm_dist(x) for x in contract.frameworks_out_of_scope}
        third = set(facts.get("third_party_imports") or {}) | declared
        hit = sorted(third & out_scope)
        if hit:
            ev = hit[0]
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=f"hors périmètre : {hit}"),
                _issue(s, f"Framework hors périmètre du profil : {hit}", evidence=ev))
        else:
            add(_mk(s, CheckStatus.PASS, note="aucun framework hors périmètre"))

    # --- ML.NON_INTERACTIVE
    if (s := spec("ML.NON_INTERACTIVE")):
        hits = facts.get("interactive_input") or []
        if hits:
            ev = j(hits[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=hits[0].get("file"), line=hits[0].get("line")),
                _issue(s, "Entrée interactive (input/stdin) détectée.", evidence=ev, file=hits[0].get("file")))
        else:
            add(_mk(s, CheckStatus.PASS))

    # --- DATA.USABLE
    if (s := spec("DATA.USABLE")):
        path = out.spec.data_path if out.spec else None
        if not path:
            add(_mk(s, CheckStatus.NOT_CHECKED, note="chemin de données inconnu (pas de manifeste exploitable)", origin="validator"))
        elif path not in files:
            ev = f"absent:{path}"
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=path),
                _issue(s, f"Fichier de données introuvable : {path}", evidence=ev, file=path))
        else:
            raw = tb.call("inspect_data", {"path": path, **({"target": out.spec.target} if out.spec and out.spec.target else {})})
            prof = json.loads(raw)
            if not prof.get("readable"):
                ev = str(prof.get("problem") or prof.get("limit") or prof.get("error") or raw)[:180]
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=path),
                    _issue(s, f"Données illisibles : {ev}", evidence=ev, file=path))
            elif prof.get("rows", 0) <= 0:
                ev = f"rows={prof.get('rows')}"
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=path), _issue(s, "Fichier de données vide.", evidence=ev, file=path))
            elif out.spec.target and isinstance(prof.get("target"), dict) and not prof["target"].get("present"):
                ev = f"target={out.spec.target}"
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=path),
                    _issue(s, f"Colonne cible absente : {out.spec.target}", evidence=ev, file=path))
            else:
                add(_mk(s, CheckStatus.PASS, file=path, note=f"{prof.get('rows')} lignes"))

    # --- SEC.*
    if (s := spec("SEC.DANGEROUS_CALLS")):
        hits = facts.get("dangerous_calls") or []
        if hits:
            ev = j(hits[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=hits[0].get("file"), line=hits[0].get("line")),
                _issue(s, f"Appel dangereux : {hits[0].get('call')}", evidence=ev, file=hits[0].get("file")))
        else:
            add(_mk(s, CheckStatus.PASS))

    if (s := spec("SEC.SECRETS")):
        hits = (facts.get("secrets") or []) + [{"file": x} for x in (facts.get("sensitive_files") or [])]
        if hits:
            ev = j(hits[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=hits[0].get("file")),
                _issue(s, "Secret ou fichier sensible détecté.", evidence=ev, file=hits[0].get("file")))
        else:
            add(_mk(s, CheckStatus.PASS))

    if (s := spec("SEC.NETWORK")):
        hits = (facts.get("network_imports") or []) + (facts.get("network_urls") or [])
        if hits:
            ev = j(hits[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=hits[0].get("file")),
                _issue(s, "Accès réseau détecté dans le code.", evidence=ev, file=hits[0].get("file")))
        else:
            add(_mk(s, CheckStatus.PASS))

    if (s := spec("SEC.COMMITTED_ARTIFACTS")):
        hits = facts.get("committed_artifacts") or []
        if hits:
            ev = str(hits[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=str(hits[0])),
                _issue(s, f"Artefact binaire versionné : {hits[0]}", evidence=ev, file=str(hits[0])))
        else:
            add(_mk(s, CheckStatus.PASS))

    # --- ML interfaces (statique : signatures) ; le harnais peut affiner ensuite
    fns = facts.get("functions") or {}
    if (s := spec("ML.TRAIN_INTERFACE")):
        e, pr = eps.entries.get("train"), eps.problem("train")
        if pr and pr.kind == "ambiguous":
            ev = ", ".join(pr.candidates[:4])
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator", note=f"fonction d'entraînement ambiguë : {pr.detail}"))
        elif e is None:
            f = (pr.file if pr and pr.file else DEFAULT_FILE["train"])
            if f not in files:
                ev = f"absent:{f}"
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f), _issue(s, f"{f} absent.", evidence=ev, file=f))
            else:
                ev = j(fns.get(f) or [])
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f),
                    _issue(s, f"{f} n'expose pas de fonction train(data_path, output_dir) ({pr.detail if pr else 'introuvable'}).", evidence=ev, file=f))
        else:
            params = e.params
            named = "data_path" in params and "output_dir" in params
            declared_ok = e.source == "manifest" and ("*" in params or len(params) >= 2)
            if named or declared_ok:
                add(_mk(s, CheckStatus.PASS, file=e.file, note=f"{e.function}{tuple(params)} résolu par {e.source}"))
            else:
                ev = j({"name": e.function, "params": params, "line": e.line})
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=e.file, line=e.line),
                    _issue(s, f"{e.function}() doit exposer data_path et output_dir (params={params}).", evidence=ev, file=e.file,
                           suggested=f"def {e.function}(data_path, output_dir): ..."))

    if (s := spec("ML.PREDICT_INTERFACE")):
        le, pe = eps.entries.get("load_model"), eps.entries.get("predict")
        amb = [p for p in eps.problems if p.kind == "ambiguous" and p.role in ("load_model", "predict")]
        if amb:
            ev = ", ".join(amb[0].candidates[:4])
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator", note=f"couple load_model/predict ambigu : {amb[0].detail}"))
        elif le and pe:
            add(_mk(s, CheckStatus.PASS, file=pe.file, note=f"load_model={le.file}:{le.function} · predict={pe.file}:{pe.function} ({pe.source})"))
        else:
            pr = eps.problem("predict") or eps.problem("load_model")
            f = (pr.file if pr and pr.file else DEFAULT_FILE["predict"])
            if f not in files:
                ev = f"absent:{f}"
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f), _issue(s, f"{f} absent.", evidence=ev, file=f))
            else:
                ev = j(fns.get(f) or [])
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f), _issue(s, "Le module de prédiction doit exposer load_model et predict.", evidence=ev, file=f))

    # --- ML.METRICS_GENUINE : ce que la fonction d'entraînement RÉSOLUE renvoie (un import sklearn.metrics ne prouve rien)
    if (s := spec("ML.METRICS_GENUINE")):
        te = eps.entries.get("train")
        tfile, tfn = (te.file, te.function) if te else (DEFAULT_FILE["train"], "train")
        p = tb.root / tfile
        text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        status, ev, note = inspect_train_metrics(text, tfn) if text else (
            CheckStatus.NOT_CHECKED, None, f"{tfile} absent ou entrypoint d'entraînement non résolu")
        if ev:
            tb.observe(ev)
        origin = "candidate" if status == CheckStatus.FAIL else "validator"
        add(_mk(s, status, evidence=ev, file=tfile if text else None, note=note, origin=origin),
            _issue(s, "Les métriques renvoyées par l'entraînement sont des littéraux, pas un calcul sur des données.",
                   evidence=ev, file=tfile) if status == CheckStatus.FAIL else None)

    # --- ML.REPRODUCIBILITY : graines = signatures inspectées dans le venv (introspection) ; déterminisme MESURÉ par le harnais
    if (s := spec("ML.REPRODUCIBILITY")):
        add(_mk(s, CheckStatus.NOT_CHECKED, note="graines : signatures inspectées dans le venv, puis déterminisme mesuré par deux entraînements ; non encore prouvé",
                origin="validator"))

    # --- IO.PATHS (statique)
    if (s := spec("IO.PATHS")):
        hits = facts.get("absolute_paths") or []
        if hits:
            ev = j(hits[0])
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, file=hits[0].get("file"), line=hits[0].get("line")),
                _issue(s, f"Chemin absolu en dur : {hits[0].get('value')}", evidence=ev, file=hits[0].get("file"),
                       suggested="Utiliser un chemin relatif ou data_path/output_dir."))
        else:
            add(_mk(s, CheckStatus.NOT_CHECKED, note="pas de chemin absolu en statique ; écritures à confirmer par le harnais",
                    origin="validator"))

    blocking = [c for c in out.checks if c.check_id in BLOCKING_SEC and c.status == CheckStatus.FAIL]
    out.security_blocked = bool(blocking) or bool(unsafe_to_execute(facts))
    return out


def apply_harness(tb, contract: AdmissionContract, outcome: DeterministicOutcome, *, auto_execute: bool) -> DeterministicOutcome:
    outcome = _apply_harness_core(tb, contract, outcome, auto_execute=auto_execute)
    outcome = apply_environment_check(tb, contract, outcome)
    outcome = apply_introspection(tb, contract, outcome, auto_execute=auto_execute)
    return collapse_cascades(contract, outcome)


def _apply_metric_probe(tb, contract: AdmissionContract, outcome: DeterministicOutcome, res: dict, train_file: str) -> None:
    """Verdict MESURÉ de ML.METRICS_GENUINE (sonde sur étiquettes permutées) : il prime sur la lecture statique du code de train()."""
    s = contract.check("ML.METRICS_GENUINE")
    mp = res.get("metric_probe") or {}
    verdict = mp.get("verdict")
    if s is None or verdict is None:
        return
    if verdict == "inconclusive":
        outcome.limits.append(f"sonde de métrique non concluante : {mp.get('why')}")
        return
    cur = next((c for c in outcome.checks if c.check_id == s.id), None)
    ev, why = mp.get("evidence"), str(mp.get("why") or "")
    if ev:
        tb.observe(ev)

    def put(chk, issue=None):
        outcome.checks = [c for c in outcome.checks if c.check_id != s.id] + [chk]
        outcome.issues = [i for i in outcome.issues if i.check_id != s.id]
        if issue is not None and chk.status == CheckStatus.FAIL:
            outcome.issues.append(issue)

    if verdict in ("label_independent", "leak"):
        fix = ("Calculer la métrique sur un jeu de validation que fit n'a pas vu (ex. train_test_split) puis la renvoyer depuis train()."
               if verdict == "leak" else
               "train() doit renvoyer la métrique calculée à partir du fichier data_path fourni (ni constante, ni valeur lue d'un fichier).")
        put(_mk(s, CheckStatus.FAIL, evidence=ev, file=train_file, note="constaté par la sonde (étiquettes permutées)", source="harness", origin="candidate"),
            _issue(s, why[:1].upper() + why[1:] + ".", evidence=ev, file=train_file, suggested=fix))
    elif verdict == "weak":
        if cur is None or cur.status != CheckStatus.FAIL:
            put(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, file=train_file, note=why, source="harness", origin="validator"))
    elif verdict == "genuine" and (cur is None or cur.status != CheckStatus.FAIL):     # une valeur littérale détectée en statique reste un FAIL
        put(_mk(s, CheckStatus.PASS, file=train_file, note=why, source="harness"))


def _apply_audit(tb, contract: AdmissionContract, outcome: DeterministicOutcome, res: dict) -> None:
    """Hook d'audit : ce que le code a RÉELLEMENT tenté pendant l'exécution, même dissimulé à l'AST (getattr, concaténation, exec).
    Seul un appelant « candidat » est sanctionné ; un lancement de processus par une bibliothèque (joblib/loky -> lscpu) est signalé."""
    find = audit_findings(res.get("audit"))

    def fail(cid, events, what):
        s = contract.check(cid)
        if s is None or not events:
            return
        ev = "; ".join(describe_event(e) for e in events[:3])
        tb.observe(ev)
        file = str(events[0].get("where") or "").split(":")[0] or None
        outcome.checks = [c for c in outcome.checks if c.check_id != s.id] + [
            _mk(s, CheckStatus.FAIL, evidence=ev, file=file, note="constaté à l'exécution (hook d'audit)", source="harness", origin="candidate")]
        outcome.issues = [i for i in outcome.issues if i.check_id != s.id] + [_issue(s, what, evidence=ev, file=file)]

    fail("SEC.NETWORK", find["network"], "Tentative d'accès réseau constatée à l'exécution.")
    fail("SEC.DANGEROUS_CALLS", find["process"], "Lancement de processus constaté à l'exécution.")
    if find["library"]:
        outcome.limits.append(f"{len(find['library'])} évènement(s) réseau/processus déclenchés par des bibliothèques (non sanctionnés) : "
                              + "; ".join(describe_event(e) for e in find["library"][:2]))


def apply_introspection(tb, contract: AdmissionContract, outcome: DeterministicOutcome, *, auto_execute: bool) -> DeterministicOutcome:
    """Ce que le venv du projet dit RÉELLEMENT (paquets installés, signatures) remplace les tables figées :
    import -> paquet déclaré, appels sans graine, estimateurs hors des listes du contrat. Rien n'est exécuté si la porte de sécurité a bloqué."""
    if not auto_execute or outcome.security_blocked:
        return outcome
    facts = tb.facts()
    req = build_request(facts)
    runner = getattr(tb, "runner", None)
    if req is None or runner is None:
        return outcome
    try:
        rep = runner.run("introspect", **req)
    except Exception as e:                      # jamais d'exception : un échec d'infra devient une limite, pas un verdict
        rep = {"not_run": True, "error": f"{type(e).__name__}: {str(e)[:100]}"}
    if not isinstance(rep, dict) or not rep.get("introspected"):
        if isinstance(rep, dict) and rep.get("error"):
            outcome.limits.append(f"introspection du venv indisponible ({str(rep['error'])[:120]}) : import→paquet, graines et frameworks à confirmer")
        return outcome
    tb.observe(j({"modules": rep.get("modules"), "symboles_résolus": sorted(rep.get("symbols") or {})[:60]}))
    mods, calls = facts.get("_third_party_modules") or {}, facts.get("_third_party_calls") or []
    declared, _ = _deps(tb)

    def current(cid):
        return next((c for c in outcome.checks if c.check_id == cid), None)

    def put(chk, issue=None):
        outcome.checks = [c for c in outcome.checks if c.check_id != chk.check_id] + [chk]
        outcome.issues = [i for i in outcome.issues if i.check_id != chk.check_id]
        if issue is not None and chk.status == CheckStatus.FAIL:
            outcome.issues.append(issue)

    # --- ENV.DEPENDENCIES_DECLARED : le venv est construit UNIQUEMENT depuis les dépendances déclarées
    if (s := contract.check("ENV.DEPENDENCIES_DECLARED")) is not None and mods:
        undeclared, unresolved = import_status(mods, rep, declared)
        absent = " ".join(c.evidence or "" for c in outcome.checks if c.check_id == "STRUCT.REQUIRED_FILES" and c.status == CheckStatus.FAIL)
        missing_local = {Path(n).stem for n in re.findall(r"absent:([\w./-]+\.py)", absent)}
        unresolved = [m for m in unresolved if m not in missing_local]     # import d'un fichier du PROJET qui manque : conséquence, pas une dépendance non déclarée
        bad = ([f"{u['module']} (installé via {', '.join(u['dists'])}, non déclaré)" for u in undeclared]
               + [f"{m} (absent de l'environnement construit depuis les dépendances déclarées)" for m in unresolved])
        if bad:
            ev = "; ".join(bad[:6])
            tb.observe(ev)
            put(_mk(s, CheckStatus.FAIL, evidence=ev, source="harness", note="imports tiers sans dépendance déclarée (introspection du venv)"),
                _issue(s, f"Imports non déclarés : {[b.split(' ')[0] for b in bad][:8]}", evidence=ev,
                       suggested="Ajouter le paquet épinglé dans requirements.txt."))
        else:
            put(_mk(s, CheckStatus.PASS, source="harness", note=f"{len(mods)} imports tiers résolus vers des paquets déclarés (introspection du venv)"))

    # --- ML.REPRODUCIBILITY : une PRÉSOMPTION par signature ; un verdict mesuré (deux entraînements) prime toujours
    if (s := contract.check("ML.REPRODUCIBILITY")) is not None:
        cur = current(s.id)
        sus = [] if (cur is not None and cur.source == "harness") else seed_suspects(calls, rep, bool(facts.get("_global_seed")))
        if sus:
            ev = j(sus[0])
            tb.observe(ev)
            put(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, file=sus[0]["file"], line=sus[0]["line"], origin="validator",
                    note=f"graine non fixée : {sus[0]['call']}({sus[0]['param']}=None par défaut), {len(sus)} appel(s) ; déterminisme non prouvé"))

    # --- FRAMEWORK.SCOPE : périmètre par paquets installés ; un estimateur inconnu du contrat n'est jamais un PASS silencieux
    if (s := contract.check("FRAMEWORK.SCOPE")) is not None:
        out_n = {norm_dist(x) for x in contract.frameworks_out_of_scope}
        dists = {d for m in mods for d in module_dists(rep, m)} | declared
        hit = sorted(dists & out_n)
        unk = {} if hit else unknown_estimator_frameworks(calls, rep, contract.frameworks_in_scope, contract.frameworks_out_of_scope)
        if hit:
            tb.observe(hit[0])
            put(_mk(s, CheckStatus.FAIL, evidence=hit[0], source="harness", note=f"hors périmètre : {hit}"),
                _issue(s, f"Framework hors périmètre du profil : {hit}", evidence=hit[0]))
        elif unk:
            ev = "; ".join(f"{d}: {', '.join(v)}" for d, v in unk.items())
            tb.observe(ev)
            put(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator",
                    note=f"estimateur d'un paquet que le contrat ne classe ni dans, ni hors du périmètre : {sorted(unk)} → revue humaine"))
        else:
            put(_mk(s, CheckStatus.PASS, source="harness", note="frameworks classés par introspection : aucun hors périmètre, aucun estimateur inconnu"))
    return outcome


def collapse_cascades(contract: AdmissionContract, outcome: DeterministicOutcome) -> DeterministicOutcome:
    """Une cause racine = un problème. Un contrôle en FAIL qui n'est que la CONSÉQUENCE d'un autre FAIL (relation `depends_on` du contrat)
    devient NOT_CHECKED (`caused_by` = racines) et perd son problème : il sera revérifié quand la racine sera corrigée.
    Un dépendant qui désigne un fichier (`file`) n'est une conséquence que si la preuve de la racine mentionne ce fichier
    (predict.py absent -> ML.PREDICT_INTERFACE ; une mauvaise signature de train() reste un problème distinct). Aucune règle de projet ici."""
    failing = {c.check_id: c for c in outcome.checks if c.status == CheckStatus.FAIL}
    demoted: dict[str, list[str]] = {}
    for c in outcome.checks:
        spec = contract.check(c.check_id)
        if c.check_id not in failing or spec is None:
            continue
        roots = []
        for d in spec.depends_on or []:
            r = failing.get(d)
            if r is None or d == c.check_id:
                continue
            if c.file and not (c.file in (r.evidence or "") or c.file == r.file):
                continue
            roots.append(d)
        if roots:
            demoted[c.check_id] = roots
    if not demoted or not (set(failing) - set(demoted)):        # cycle de dépendances : on ne masque rien
        return outcome
    for c in outcome.checks:
        if c.check_id in demoted:
            roots = demoted[c.check_id]
            c.note = f"conséquence de {', '.join(roots)} : revérifier après correction ({(c.note or c.evidence or '')[:100]})".strip()
            c.status, c.origin, c.caused_by = CheckStatus.NOT_CHECKED, "validator", roots
    outcome.issues = [i for i in outcome.issues if i.check_id not in demoted]
    outcome.limits.append("contrôles en cascade non comptés comme problèmes distincts : " + ", ".join(f"{k} (← {', '.join(v)})" for k, v in demoted.items()))
    return outcome


def apply_environment_check(tb, contract: AdmissionContract, outcome: DeterministicOutcome) -> DeterministicOutcome:
    """ENV.SANDBOX_ISOLATION : le projet a-t-il tourné dans SON environnement isolé, avec SES dépendances ? Python seul, sur des faits."""
    s = contract.check("ENV.SANDBOX_ISOLATION")
    if s is None:
        return outcome
    env = (tb.sandbox_info or {}).get("environment")
    if not env:
        chk = _mk(s, CheckStatus.NOT_CHECKED, note="aucune exécution dans le sandbox : environnement non vérifiable", origin="validator", source="python")
        issue = None
    else:
        mode, bad = env.get("mode"), (env.get("mismatches") or []) + [{"missing": m} for m in env.get("missing") or []]
        summary = (f"mode={mode} · backend={env.get('backend')} · python={env.get('python')} · venv={env.get('is_virtualenv')} · "
                   f"dépendances {env.get('deps_source') or 'aucune'} · {env.get('pinned_checked', 0)} épinglée(s) comparée(s) · "
                   f"réseau isolé={env.get('network_isolated')}")
        tb.observe(summary)
        if not env.get("network_isolated") and env.get("backend") != "docker":
            outcome.limits.append("sandbox sans isolation réseau (repli sous-processus) : le garde-fou statique reste la protection principale")
        if mode == "current" or env.get("shares_validator_env"):
            chk = _mk(s, CheckStatus.FAIL, evidence=summary, note="le projet s'exécute dans l'interpréteur du validateur, pas dans son propre environnement", origin="validator", source="python")
            issue = _issue(s, "Exécution hors environnement dédié (--env current) : dépendances du validateur utilisées.", evidence=summary)
        elif bad:
            ev = summary + " · écarts : " + json.dumps(bad[:5], ensure_ascii=False)
            tb.observe(ev)
            chk = _mk(s, CheckStatus.FAIL, evidence=ev, note="l'environnement ne contient pas les versions épinglées du projet", origin="harness", source="python")
            issue = _issue(s, "L'environnement d'exécution ne correspond pas aux dépendances épinglées du projet.", evidence=ev)
        elif env.get("backend") == "docker":
            chk = _mk(s, CheckStatus.PASS, evidence=summary, note="image Docker construite pour le projet (environnement non inspecté depuis l'hôte)", source="python")
            issue = None
        elif env.get("own_venv") or (mode == "custom" and env.get("is_virtualenv") and env.get("verified") and env.get("pinned_checked")):
            note = ("venv dédié construit pour ce projet" if env.get("own_venv") else "interpréteur fourni (--env) conforme aux dépendances épinglées du projet")
            chk = _mk(s, CheckStatus.PASS, evidence=summary, note=note, source="python")
            issue = None
        else:
            chk = _mk(s, CheckStatus.NOT_CHECKED, evidence=summary, note=env.get("note") or "environnement non vérifiable", origin="validator", source="python")
            issue = None
    outcome.checks = [c for c in outcome.checks if c.check_id != s.id] + [chk]
    if issue and chk.status == CheckStatus.FAIL:
        outcome.issues = [i for i in outcome.issues if i.check_id != s.id] + [issue]
    return outcome


def _apply_harness_core(tb, contract: AdmissionContract, outcome: DeterministicOutcome, *, auto_execute: bool) -> DeterministicOutcome:
    """Porte de sécurité puis harnais. Un échec d'infra ne devient pas un FAIL candidat."""
    def spec(cid):
        return contract.check(cid)

    def upsert(chk: Check, issue: Issue | None = None):
        outcome.checks = [c for c in outcome.checks if c.check_id != chk.check_id] + [chk]
        if issue and chk.status == CheckStatus.FAIL:
            if issue.evidence and not tb.is_grounded(issue.evidence):
                tb.observe(issue.evidence)
            outcome.issues = [i for i in outcome.issues if i.check_id != issue.check_id] + [issue]

    if spec("EXEC.RUN") is None:
        return outcome

    s_exec = spec("EXEC.RUN")
    if not auto_execute:
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED, note="exécution non lancée par Python (auto_execute=false)",
                   origin="validator", source="python"))
        return outcome

    if outcome.security_blocked:
        ev = "exécution refusée : faits statiques dangereux"
        tb.observe(ev)
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED, evidence=ev,
                   note="porte de sécurité : EXEC.* non lancés", origin="validator"))
        for cid in ("ML.TRAIN_INTERFACE", "ML.PREDICT_INTERFACE", "ML.REPRODUCIBILITY", "IO.PATHS"):
            cur = next((c for c in outcome.checks if c.check_id == cid), None)
            if cur and cur.status == CheckStatus.NOT_CHECKED:
                cur.note = (cur.note or "") + " ; harnais non lancé (sécurité)"
        outcome.executed_ok = False
        outcome.limits.append("exécution non lancée : contrôle de sécurité bloquant")
        return outcome

    spec_p = outcome.spec
    if not spec_p or not (spec_p.data_path and spec_p.target):
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED,
                   note="harnais impossible : data_path ou target inconnus (ni manifeste, ni déduction vérifiable)", origin="validator"))
        outcome.limits.append("harnais non lancé : spécification projet incomplète")
        return outcome

    raw = tb.call("run_in_sandbox", {"command": "harness", "data_path": spec_p.data_path, "target": spec_p.target,
                                     **({"primary_metric": spec_p.primary_metric} if spec_p.primary_metric else {})})
    try:
        res = json.loads(raw)
    except Exception:
        res = {"error": raw[:200], "not_run": True}

    declared, _ = _deps(tb)
    origin = classify_exec_error(res, env=getattr(tb.runner, "env", "venv") or "venv", declared=declared)

    if res.get("refused"):
        ev = str(res.get("reason") or raw)[:180]
        tb.observe(ev)
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED, evidence=ev, note=ev, origin="validator"))
        outcome.executed_ok = False
        return outcome

    if res.get("not_run"):
        ev = str(res.get("error") or raw)[:180]
        tb.observe(ev)
        outcome.limits.append(f"exécution impossible ({origin}) : {ev}")
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED, evidence=ev,
                   note=f"problème {origin}, pas un verdict sur le projet" if origin != "candidate" else ev,
                   origin=origin, source="harness"))
        outcome.executed_ok = False
        return outcome

    ok = bool(res.get("executed_ok"))
    outcome.executed_ok = ok
    md = res.get("metric_discovery")
    if md and md.get("chosen") and not spec_p.primary_metric:       # métrique confirmée par l'exécution réelle de train()
        hint = (getattr(tb, "hints", None) or {}).get("metric_hint")
        if hint and hint in (md.get("candidates") or []):
            spec_p.primary_metric = hint
            spec_p.deduced["primary_metric"] = "proposée par le LLM ; clé effectivement présente dans le dict renvoyé par train()"
        else:
            spec_p.primary_metric = md["chosen"]
            spec_p.deduced["primary_metric"] = f"clé du dict renvoyé par train() à l'exécution : {md['rule']}"
    ev = res.get("_summary") or ("exécution réussie" if ok else "exécution NON réussie")
    tb.observe(str(ev))
    if ok:
        upsert(_mk(s_exec, CheckStatus.PASS, evidence=None, note=ev, source="harness", origin="candidate"))
    else:
        upsert(_mk(s_exec, CheckStatus.FAIL if origin == "candidate" else CheckStatus.NOT_CHECKED,
                   evidence=str(ev), note=f"origin={origin}", source="harness", origin=origin),
               _issue(s_exec, f"Exécution sandbox échouée ({origin}).", evidence=str(ev)) if origin == "candidate" else None)

    steps = res.get("contract_steps") or {}
    ep_files = {r: str(v).partition(":")[0] for r, v in (res.get("entrypoints") or {}).items()}
    train_file, predict_file = ep_files.get("train", "train.py"), ep_files.get("predict", "predict.py")
    if (s := spec("ML.TRAIN_INTERFACE")):
        cur = next((c for c in outcome.checks if c.check_id == s.id), None)
        train_ok = steps.get("import_train") == "ok" and steps.get("train") == "ok"
        if ok and train_ok and (cur is None or cur.status != CheckStatus.FAIL):
            upsert(_mk(s, CheckStatus.PASS, source="harness", file=train_file))
        elif not ok and steps.get("train", "").startswith("FAILED"):
            evs = steps["train"][:180]
            tb.observe(evs)
            if origin == "candidate":
                upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness", file=train_file),
                       _issue(s, "train() a échoué dans le harnais.", evidence=evs, file=train_file))

    if (s := spec("ML.PREDICT_INTERFACE")):
        if contract.harness == "sklearn_tabular":
            pred_ok = all(steps.get(k) == "ok" for k in ("import_predict", "load_model", "predict_df", "predict_dict"))
            if ok and pred_ok:
                upsert(_mk(s, CheckStatus.PASS, source="harness", file=predict_file))
            elif any(str(steps.get(k, "")).startswith("FAILED") for k in ("import_predict", "load_model", "predict_df", "predict_dict")) and origin == "candidate":
                evs = j({k: steps.get(k) for k in ("import_predict", "load_model", "predict_df", "predict_dict")})
                tb.observe(evs)
                upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness", file=predict_file),
                       _issue(s, "Interface predict échouée dans le harnais.", evidence=evs, file=predict_file))

    if (s := spec("ML.REPRODUCIBILITY")):
        det = res.get("deterministic")
        if det is True:
            upsert(_mk(s, CheckStatus.PASS, source="harness", note="deux runs identiques"))
        elif det is False:
            evs = j(res.get("metric_differences") or {"deterministic": False})
            tb.observe(evs)
            upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness"),
                   _issue(s, "Deux entraînements ne reproduisent pas les mêmes métriques.", evidence=evs))

    if (s := spec("IO.PATHS")):
        outside = res.get("files_written_outside_output_dir") or []
        cur = next((c for c in outcome.checks if c.check_id == s.id), None)
        if outside:
            evs = j(outside[:8])
            tb.observe(evs)
            upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness"),
                   _issue(s, f"Écritures hors output_dir : {outside[:8]}", evidence=evs))
        elif cur and cur.status == CheckStatus.NOT_CHECKED and (ok or steps.get("train") == "ok"):
            # les écritures sont mesurées sur main.py + harnais : inutile d'attendre que TOUT réussisse (ex. predict.py absent)
            upsert(_mk(s, CheckStatus.PASS, source="harness", note="pas d'écriture hors output_dir (train() exécuté)"))

    _apply_metric_probe(tb, contract, outcome, res, train_file)
    _apply_audit(tb, contract, outcome, res)
    return outcome


def dump_outcome(out: DeterministicOutcome) -> dict:
    return {"checks": [c.model_dump(mode="json") for c in out.checks],
            "issues": [i.model_dump(mode="json") for i in out.issues],
            "spec": out.spec.model_dump(mode="json") if out.spec else None,
            "limits": list(out.limits), "security_blocked": out.security_blocked,
            "executed_ok": out.executed_ok, "entrypoints": out.entrypoints}


def load_outcome(data: dict) -> DeterministicOutcome:
    spec = data.get("spec")
    return DeterministicOutcome(
        checks=[Check.model_validate(c) for c in data.get("checks") or []],
        issues=[Issue.model_validate(i) for i in data.get("issues") or []],
        spec=ProjectSpec.model_validate(spec) if spec else None,
        limits=list(data.get("limits") or []),
        security_blocked=bool(data.get("security_blocked")),
        executed_ok=data.get("executed_ok"), entrypoints=data.get("entrypoints"))


def deterministic_cache_key(project_hash: str, contract: AdmissionContract, *, env: str, sandbox: str, auto_execute: bool) -> str:
    raw = f"{project_hash}\n{contract.id}@{contract.version}\n{env}\n{sandbox}\n{int(auto_execute)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def load_deterministic_cache(path: Path, *, project_hash: str, contract: AdmissionContract) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("project_hash") != project_hash or data.get("contract") != f"{contract.id}@{contract.version}":
        return None
    return data


def save_deterministic_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


PYTHON_ONLY_CHECKS = {"ENV.SANDBOX_ISOLATION"}      # jamais décidés ni modifiés par le LLM


def _llm_remark(c: Check) -> str:
    return f"LLM ({c.status.value})" + (f" : {c.note}" if c.note else "")


def _capped(c: Check, *, allow_warning: bool) -> Check:
    """Un contrôle OBLIGATOIRE ne reçoit jamais de PASS du LLM, ni de FAIL : au pire un avertissement (si Python l'avait déjà validé)
    ou NOT_CHECKED (-> NEEDS_REVIEW). Un FAIL d'origine LLM n'a pas de fait Python derrière lui : il ouvre une revue, il ne condamne pas."""
    if c.status == CheckStatus.NOT_CHECKED or (c.status == CheckStatus.WARNING and allow_warning):
        return c
    why = {CheckStatus.PASS: "PASS du LLM non retenu : contrôle obligatoire, sans preuve Python",
           CheckStatus.NOT_APPLICABLE: "NOT_APPLICABLE du LLM non retenu : contrôle obligatoire",
           CheckStatus.FAIL: "le LLM soupçonne un défaut non confirmé par Python → revue humaine",
           CheckStatus.WARNING: "avertissement du LLM sur un contrôle que Python n'a pas validé → revue humaine"}[c.status]
    return c.model_copy(update={"status": CheckStatus.NOT_CHECKED, "evidence": None, "origin": "validator",
                                "note": (why + (f" · {c.note}" if c.note else ""))[:400]})


def merge_llm_over_python(python_checks: list[Check], llm_checks: list[Check], mandatory: frozenset = frozenset()) -> list[Check]:
    """Le LLM ne peut pas annuler un FAIL Python, ni transformer NOT_CHECKED/FAIL en PASS sans preuve Python/harnais.
    `mandatory` (ids des contrôles obligatoires du contrat) : sur ces contrôles le LLM ne peut JAMAIS produire PASS ni FAIL ;
    il peut seulement laisser/ouvrir NOT_CHECKED (NEEDS_REVIEW) avec son explication, ou ajouter un WARNING sur un PASS Python
    de ML.METRICS_GENUINE (interprétation). Python est la seule source de PASS/FAIL obligatoires."""
    rank = {CheckStatus.FAIL: 4, CheckStatus.WARNING: 3, CheckStatus.NOT_CHECKED: 2,
            CheckStatus.NOT_APPLICABLE: 1, CheckStatus.PASS: 0}
    out = {c.check_id: c for c in python_checks}
    for c in llm_checks:
        if c.check_id in PYTHON_ONLY_CHECKS:
            continue
        is_mand = c.check_id in mandatory
        cur = out.get(c.check_id)
        if cur is None:
            c = _capped(c, allow_warning=False) if is_mand else c
            out[c.check_id] = c.model_copy(update={"source": c.source or "llm"})
            continue
        if cur.source in ("python", "harness") and cur.status == CheckStatus.FAIL:
            continue
        if cur.source in ("python", "harness") and cur.status == CheckStatus.PASS and c.status != CheckStatus.PASS:
            # alerte LLM : on ne dégrade un PASS mécanique que pour ML.METRICS_GENUINE (interprétation)
            if cur.check_id != "ML.METRICS_GENUINE":
                continue
        if cur.status == CheckStatus.NOT_CHECKED:
            if cur.caused_by:
                continue                # conséquence d'une cause racine en FAIL : seule la revalidation la tranche
            if c.status == CheckStatus.PASS and cur.check_id.startswith("EXEC."):
                continue
            if is_mand:                 # reste ouvert ; l'avis du LLM éclaire le relecteur, il ne décide rien
                out[c.check_id] = cur.model_copy(update={"note": ((cur.note or "") + " · " + _llm_remark(c)).strip(" ·")[:500]})
                continue
            out[c.check_id] = c.model_copy(update={"source": c.source or "llm"})
            continue
        if rank[c.status] > rank[cur.status] and cur.check_id == "ML.METRICS_GENUINE":
            c = _capped(c, allow_warning=True) if is_mand else c
            out[c.check_id] = c.model_copy(update={"source": c.source or "llm"})
    return list(out.values())