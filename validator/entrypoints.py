"""Résolution des points d'entrée du projet : manifeste d'abord, AST ensuite, convention en dernier recours.

Rien ici n'exécute le code du candidat : on lit le manifeste et on parse les sources (`ast`).
Le résultat (`Resolution`) sert à trois endroits :
  - les contrôles statiques (fichiers requis déduits, interfaces train/predict) ;
  - le harnais d'exécution (il reçoit des `module:fonction` déjà résolus, plus le `sys_path` à ajouter) ;
  - le Repair (même harnais, donc même résolution).

Rôles : `train`, `load_model`, `predict`.
Formes acceptées dans `entrypoints:` du manifeste (valeurs de `train`, `predict`, `load_model`, `main`) :
    train.py                 fichier, fonction par défaut du rôle
    train.py:fit             fichier + fonction
    src/pkg/train.py:fit     chemin relatif + fonction
    pkg.train:fit            module pointé + fonction (cherché dans le projet, `src/` compris)
    train:fit                idem
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

from .fs import read_text_safe

ROLES = ("train", "load_model", "predict")
DEFAULT_FUNCTION = {"train": "train", "load_model": "load_model", "predict": "predict"}
DEFAULT_FILE = {"train": "train.py", "load_model": "predict.py", "predict": "predict.py"}
ROLES_BY_HARNESS = {"sklearn_tabular": ("train", "load_model", "predict"), "train_only": ("train",)}

TRAIN_PARAMS = ("data_path", "output_dir")
NAME_HINT_FILES = {"train": ("train", "training", "fit"), "predict": ("predict", "inference", "serve", "predictor")}
NAME_HINT_FUNCS = ("train", "fit")
NON_PROJECT_PARTS = {"tests", "test", "docs", "examples", "scripts", "migrations"}


@dataclass
class Entry:
    role: str
    file: str                      # chemin relatif posix
    function: str
    module: str | None             # nom importable, None si le chemin n'est pas importable (loader par fichier)
    sys_path: str                  # dossier (relatif au projet) à mettre sur sys.path ; "." = racine
    source: str                    # manifest | ast | convention
    params: list[str] = field(default_factory=list)
    required: int = 0
    line: int | None = None

    def spec(self) -> str:
        return f"{self.module or self.file}:{self.function}"

    def harness(self) -> dict:
        return {"module": self.module, "function": self.function, "file": self.file, "sys_path": self.sys_path, "source": self.source}


@dataclass
class Problem:
    role: str
    kind: str                      # missing | ambiguous | invalid
    detail: str
    file: str | None = None        # fichier attendu (pour la preuve « absent:<fichier> »)
    candidates: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    entries: dict[str, Entry] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    manifest_declared: dict[str, str] = field(default_factory=dict)
    main: str | None = None
    main_declared: bool = False

    def problem(self, role: str) -> Problem | None:
        return next((p for p in self.problems if p.role == role), None)

    def ambiguous(self, roles: Iterable[str] | None = None) -> list[Problem]:
        return [p for p in self.problems if p.kind == "ambiguous" and (roles is None or p.role in roles)]

    def required_files(self, roles: Iterable[str]) -> list[str]:
        out: list[str] = []
        for r in roles:
            e, p = self.entries.get(r), self.problem(r)
            f = e.file if e else (p.file if p and p.file else DEFAULT_FILE[r])
            if p and p.kind == "ambiguous":
                continue
            if f not in out:
                out.append(f)
        if self.main_declared and self.main and self.main not in out:
            out.append(self.main)
        return out

    def harness_map(self, roles: Iterable[str]) -> dict:
        """Ce que reçoit le harnais. Rôle manquant -> convention (le harnais échouera proprement à l'import).
        Rôle ambigu -> absent : l'appelant doit refuser d'exécuter (voir `ambiguous`)."""
        m: dict = {}
        for r in roles:
            e = self.entries.get(r)
            if e is not None:
                m[r] = e.harness()
            elif not self.problem(r) or self.problem(r).kind != "ambiguous":
                f = DEFAULT_FILE[r]
                m[r] = {"module": Path(f).stem, "function": DEFAULT_FUNCTION[r], "file": f, "sys_path": ".", "source": "convention"}
        return m

    def summary(self) -> dict:
        return {"resolved": {r: f"{e.file}:{e.function} ({e.source})" for r, e in self.entries.items()},
                "problems": [{"role": p.role, "kind": p.kind, "detail": p.detail, "candidates": p.candidates[:6]} for p in self.problems],
                "main": self.main}


# ------------------------------------------------------------------ lecture des sources
def _is_project_code(rel: str) -> bool:
    p = PurePosixPath(rel)
    if p.suffix != ".py":
        return False
    if p.name.startswith("test_") or p.name.endswith("_test.py") or p.name in ("conftest.py", "setup.py"):
        return False
    return not (set(p.parts[:-1]) & NON_PROJECT_PARTS)


def _scan(root: Path, files: Iterable[str]) -> dict[str, dict]:
    """rel -> {"funcs": {nom: {params, required, line}}, "provides": {noms définis ou ré-exportés par import}}."""
    out: dict[str, dict] = {}
    for rel in files:
        if not rel.endswith(".py"):
            continue
        text = read_text_safe(root / rel)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            continue
        funcs: dict[str, dict] = {}
        provides: set[str] = set()
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pos = [a.arg for a in n.args.posonlyargs + n.args.args]
                funcs[n.name] = {"params": pos, "required": len(pos) - len(n.args.defaults), "line": n.lineno,
                                 "varargs": n.args.vararg is not None}
                provides.add(n.name)
            elif isinstance(n, ast.ImportFrom):
                provides |= {a.asname or a.name for a in n.names}
        out[rel] = {"funcs": funcs, "provides": provides}
    return out


def module_location(rel: str, filelist: set[str]) -> tuple[str | None, str]:
    """(nom de module importable, dossier à mettre sur sys.path) pour un fichier .py du projet.
    On remonte tant que le dossier parent contient un __init__.py ; le premier dossier SANS __init__.py est la racine d'import
    (donc `src/pkg/train.py` -> module `pkg.train`, sys_path `src`)."""
    p = PurePosixPath(rel)
    parts = list(p.parent.parts)
    k = len(parts)
    while k > 0 and "/".join(parts[:k] + ["__init__.py"]) in filelist:
        k -= 1
    base = parts[:k]
    mod_parts = parts[k:] + ([] if p.stem == "__init__" else [p.stem])
    sys_path = "/".join(base) or "."
    if not mod_parts or not all(x.isidentifier() for x in mod_parts):
        return None, sys_path
    return ".".join(mod_parts), sys_path


# ------------------------------------------------------------------ manifeste
def _clean_rel(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().replace("\\", "/")
    if s.startswith("/") or ".." in s.split("/") or (len(s) > 1 and s[1] == ":"):
        return None
    return str(PurePosixPath(s))


def _find_file(target: str, filelist: set[str]) -> list[str]:
    """`target` = chemin (a/b.py) ou module pointé (a.b). Retourne les fichiers du projet qui correspondent."""
    cand: list[str] = []
    if target.endswith(".py"):
        t = _clean_rel(target)
        return [t] if t and t in filelist else []
    dotted = target.replace(".", "/")
    for suffix in (f"{dotted}.py", f"{dotted}/__init__.py"):
        cand += sorted(f for f in filelist if f == suffix or f.endswith("/" + suffix))
    return cand


def _from_manifest(role: str, raw, scan: dict, filelist: set[str], res: Resolution, default_file: str | None = None) -> None:
    if not isinstance(raw, str) or not raw.strip():
        res.problems.append(Problem(role, "invalid", f"entrypoints.{role} doit être une chaîne"))
        return
    target, _, fn = raw.strip().partition(":")
    fn = fn.strip() or DEFAULT_FUNCTION[role]
    if _clean_rel(target if target.endswith(".py") else target.replace(".", "/")) is None:
        res.problems.append(Problem(role, "invalid", f"chemin d'entrypoint refusé : {raw!r}"))
        return
    found = _find_file(target, filelist)
    if not found:
        shown = _clean_rel(target) if target.endswith(".py") else target.replace(".", "/") + ".py"
        res.problems.append(Problem(role, "missing", f"fichier déclaré introuvable : {raw}", file=shown))
        return
    if len(found) > 1:
        res.problems.append(Problem(role, "ambiguous", f"plusieurs fichiers pour {target}", candidates=found))
        return
    rel = found[0]
    info = scan.get(rel)
    f = (info or {}).get("funcs", {}).get(fn)
    if f is None and info and fn in info["provides"]:      # ré-exporté par import : appelable, mais signature inconnue
        f = {"params": ["*"], "required": 0, "line": None, "varargs": True}      # "*" = signature inconnue (ré-export)
    if f is None:
        res.problems.append(Problem(role, "missing", f"fonction `{fn}` absente de {rel}", file=rel))
        return
    mod, sp = module_location(rel, filelist)
    res.entries[role] = Entry(role, rel, fn, mod, sp, "manifest", f["params"], f["required"], f["line"])


# ------------------------------------------------------------------ AST
def _pick(cands: list[tuple[str, str]], role_key: str) -> tuple[tuple[str, str] | None, list[str]]:
    """cands = [(fichier, fonction)]. Départage par nom de fichier puis par nom de fonction ; sinon ambigu."""
    if len(cands) == 1:
        return cands[0], []
    hints = NAME_HINT_FILES[role_key]
    by_file = [c for c in cands if PurePosixPath(c[0]).stem in hints]
    if len(by_file) == 1:
        return by_file[0], []
    pool = by_file or cands
    by_func = [c for c in pool if c[1] in NAME_HINT_FUNCS or c[1] in ("load_model", "predict")]
    if len(by_func) == 1:
        return by_func[0], []
    # à profondeur égale, la racine l'emporte sur un sous-dossier
    shallow = min(len(PurePosixPath(c[0]).parts) for c in pool)
    top = [c for c in pool if len(PurePosixPath(c[0]).parts) == shallow]
    if len(top) == 1:
        return top[0], []
    return None, [f"{f}:{fn}" for f, fn in cands]


def _ast_train(scan: dict, filelist: set[str], res: Resolution) -> None:
    cands = [(rel, fn) for rel, info in scan.items() if _is_project_code(rel)
             for fn, d in info["funcs"].items() if all(p in d["params"] for p in TRAIN_PARAMS)]
    if not cands:
        return
    chosen, amb = _pick(cands, "train")
    if chosen is None:
        res.problems.append(Problem("train", "ambiguous", "plusieurs fonctions ont les paramètres data_path/output_dir", candidates=amb))
        return
    rel, fn = chosen
    d = scan[rel]["funcs"][fn]
    mod, sp = module_location(rel, filelist)
    res.entries["train"] = Entry("train", rel, fn, mod, sp, "ast", d["params"], d["required"], d["line"])


def _ast_predict(scan: dict, filelist: set[str], res: Resolution, need: set[str]) -> None:
    both = [rel for rel, info in scan.items() if _is_project_code(rel)
            and {"load_model", "predict"} <= info["provides"] and "predict" in info["funcs"]]
    chosen = None
    if both:
        pick, amb = _pick([(r, "predict") for r in both], "predict")
        if pick is None:
            for r in need:
                res.problems.append(Problem(r, "ambiguous", "plusieurs modules exposent load_model + predict", candidates=amb))
            return
        chosen = pick[0]
    if chosen is None:
        return
    info = scan[chosen]
    mod, sp = module_location(chosen, filelist)
    for r in need:
        d = info["funcs"].get(r) or {"params": [], "required": 0, "line": None}
        res.entries[r] = Entry(r, chosen, DEFAULT_FUNCTION[r], mod, sp, "ast", d["params"], d["required"], d["line"])


def _convention(role: str, scan: dict, filelist: set[str], res: Resolution) -> None:
    f = DEFAULT_FILE[role]
    info = scan.get(f)
    fn = DEFAULT_FUNCTION[role]
    d = (info or {}).get("funcs", {}).get(fn)
    if d is None and info and fn in info["provides"]:
        d = {"params": [], "required": 0, "line": None}
    if d is None:
        return
    mod, sp = module_location(f, filelist)
    res.entries[role] = Entry(role, f, fn, mod, sp, "convention", d["params"], d["required"], d["line"])


# ------------------------------------------------------------------ point d'entrée public
def resolve_entrypoints(root, files: Iterable, manifest: dict | None = None, harness: str | None = "sklearn_tabular") -> Resolution:
    """`files` : chemins relatifs du projet (str ou Path). `manifest` : le CORPS du mlops.yaml (dict) ou None.
    Ordre : 1) `entrypoints` du manifeste  2) AST (signature / couple load_model+predict)  3) convention train.py / predict.py."""
    root = Path(root)
    flist = {Path(f).as_posix() for f in files}
    roles = ROLES_BY_HARNESS.get(harness or "", ROLES)
    scan = _scan(root, sorted(flist))
    res = Resolution()
    body = (manifest or {}).get("entrypoints") if isinstance(manifest, dict) else None
    body = body if isinstance(body, dict) else {}
    res.manifest_declared = {k: v for k, v in body.items() if isinstance(v, str)}

    # 1) manifeste
    declared_pred = body.get("predict")
    for role in roles:
        if role == "train" and "train" in body:
            _from_manifest("train", body["train"], scan, flist, res)
        elif role == "predict" and declared_pred is not None:
            _from_manifest("predict", declared_pred, scan, flist, res)
        elif role == "load_model":
            if "load_model" in body:
                _from_manifest("load_model", body["load_model"], scan, flist, res)
            elif declared_pred is not None and isinstance(declared_pred, str):      # load_model vit dans le même module que predict
                target = declared_pred.partition(":")[0]
                _from_manifest("load_model", f"{target}:load_model", scan, flist, res)

    # 2) AST puis 3) convention, seulement pour les rôles encore non traités
    todo = [r for r in roles if r not in res.entries and res.problem(r) is None]
    if "train" in todo:
        _ast_train(scan, flist, res)
        if "train" not in res.entries and res.problem("train") is None:
            _convention("train", scan, flist, res)
    pred_todo = {r for r in ("load_model", "predict") if r in todo}
    if pred_todo:
        _ast_predict(scan, flist, res, pred_todo)
        for r in sorted(pred_todo):
            if r not in res.entries and res.problem(r) is None:
                _convention(r, scan, flist, res)

    # rôles restés sans entrée : « manquant », avec le fichier attendu pour la preuve `absent:<fichier>`
    for r in roles:
        if r not in res.entries and res.problem(r) is None:
            f = DEFAULT_FILE[r]
            why = f"{f} absent" if f not in flist else f"fonction `{DEFAULT_FUNCTION[r]}` introuvable dans {f}"
            res.problems.append(Problem(r, "missing", why, file=f))

    # main : facultatif sauf s'il est déclaré
    m = body.get("main")
    if isinstance(m, str) and _clean_rel(m.partition(":")[0]):
        res.main, res.main_declared = _clean_rel(m.partition(":")[0]), True
    elif "main.py" in flist:
        res.main = "main.py"
    return res


def read_manifest_body(root) -> dict:
    """Corps YAML de mlops.yaml/yml ; {} si absent ou illisible (la validité du manifeste se juge ailleurs)."""
    root = Path(root)
    for name in ("mlops.yaml", "mlops.yml"):
        p = root / name
        if p.is_file():
            try:
                import yaml
                d = yaml.safe_load(p.read_text(encoding="utf-8"))
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}
