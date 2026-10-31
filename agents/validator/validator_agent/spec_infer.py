"""Déduction de la spec projet (chemin des données, cible, tâche) SANS manifeste, puis vérification par Python.

Principe « Python propose, Python vérifie » : chaque valeur déduite doit être CONFIRMÉE par une observation
  - data_path : le fichier existe dans le projet, est lisible, non vide ;
  - target    : la colonne existe dans l'en-tête du CSV ;
  - task      : cohérente avec la colonne cible (et avec les estimateurs importés, sinon on s'abstient) ;
  - primary_metric : n'est PAS devinée ici : elle est lue dans le dict renvoyé par l'entraînement réel (voir SandboxRunner._harness).
Les valeurs du manifeste gagnent toujours ; on ne comble que les trous. En cas d'ambiguïté : rien n'est inventé (le contrôle reste ouvert).
Chaque valeur déduite est tracée dans `ProjectSpec.deduced` (slot -> preuve).
"""
from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

from agentic_core.schemas import ProjectSpec

DATA_EXT = (".csv", ".tsv")
EXCLUDED_DIRS = {"outputs", "output", "models", "model", "artifacts", "results", "reports", "tests", "test", "docs", "examples", "node_modules"}
TARGET_NAME = re.compile(r"(target|label)", re.I)
Y_VAR = re.compile(r"^(y|y_all|y_full|labels?)$", re.I)
X_VAR = re.compile(r"^(x|x_all|x_full|features?)$", re.I)
CONVENTIONAL_COLUMNS = ("target", "label", "y", "class")
CLASSIF_HINT = re.compile(r"(Classifier|Classification|LogisticRegression|^SVC$)")
REGRESS_HINT = re.compile(r"(Regressor|^LinearRegression$|^Ridge|^Lasso|^ElasticNet|^SVR$)")


# ------------------------------------------------------------------ lecture des sources
def _parse(root, rel):
    try:
        return ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"), filename=rel)
    except (SyntaxError, OSError):
        return None


def _scope_files(entry_file: str | None, files: list[str]) -> list[str]:
    """Fichiers qui décrivent l'entraînement : l'entrypoint et ses voisins directs (même dossier)."""
    if not entry_file:
        return []
    d = PurePosixPath(entry_file).parent
    return [f for f in files if f.endswith(".py") and PurePosixPath(f).parent == d]


def _consts(tree) -> list[tuple[str, int]]:
    return [(n.value, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _unwrap(node):
    """df["y"].astype(int) -> df["y"]"""
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not (node.func.attr in ("pop", "drop")):
        node = node.func.value
    return node


def _drop_consts(call: ast.Call) -> list[str]:
    out: list[str] = []
    vals = list(call.args[:1]) + [k.value for k in call.keywords if k.arg in ("columns", "labels")]
    for v in vals:
        if (s := _str(v)) is not None:
            out.append(s)
        elif isinstance(v, (ast.List, ast.Tuple)):
            out += [s for e in v.elts if (s := _str(e)) is not None]
    return out


def target_candidates(tree, callee_params: dict[str, list[str]]) -> dict[str, list[tuple[str, int]]]:
    """Contextes syntaxiques où une constante désigne la cible, du plus fiable (1) au moins fiable (3)."""
    t1: list = []
    t2: list = []
    t3: list = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            val = n.value
            if len(tgts) != 1 or not isinstance(tgts[0], ast.Name) or val is None:
                continue
            name = tgts[0].id
            if (s := _str(val)) is not None and TARGET_NAME.search(name):
                t1.append((s, n.lineno))
            v = _unwrap(val)
            if Y_VAR.match(name):
                if isinstance(v, ast.Subscript) and (s := _str(v.slice)) is not None:
                    t2.append((s, n.lineno))
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "pop" and v.args and (s := _str(v.args[0])):
                    t2.append((s, n.lineno))
            if X_VAR.match(name) and isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == "drop":
                t3 += [(c, n.lineno) for c in _drop_consts(val)]
        elif isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg and TARGET_NAME.search(kw.arg) and (s := _str(kw.value)) is not None:
                    t1.append((s, n.lineno))
            fname = n.func.id if isinstance(n.func, ast.Name) else (n.func.attr if isinstance(n.func, ast.Attribute) else None)
            params = callee_params.get(fname or "")
            if params:
                for i, a in enumerate(n.args):
                    if i < len(params) and TARGET_NAME.search(params[i]) and (s := _str(a)) is not None:
                        t1.append((s, n.lineno))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            pos = a.posonlyargs + a.args
            for arg, d in zip(reversed(pos), reversed(a.defaults)):
                if TARGET_NAME.search(arg.arg) and (s := _str(d)) is not None:
                    t1.append((s, n.lineno))
    return {"param/variable nommée target|label": t1, "y = df[...]": t2, "X = df.drop(...)": t3}


def _callee_params(root, files) -> dict[str, list[str]]:
    seen: dict[str, list[list[str]]] = {}
    for rel in files:
        if not rel.endswith(".py"):
            continue
        t = _parse(root, rel)
        for n in (t.body if t else []):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen.setdefault(n.name, []).append([a.arg for a in n.args.posonlyargs + n.args.args])
    return {k: v[0] for k, v in seen.items() if len(v) == 1}       # ambigu (plusieurs définitions) -> ignoré


# ------------------------------------------------------------------ données
def _data_files(files: list[str]) -> list[str]:
    return [f for f in files if f.lower().endswith(DATA_EXT) and not (set(PurePosixPath(f).parts[:-1]) & EXCLUDED_DIRS)]


def deduce_data_path(root, files: list[str], scope: list[str]) -> tuple[str | None, str]:
    cands = _data_files(files)
    if not cands:
        return None, "aucun fichier .csv/.tsv hors dossiers de sortie"
    referenced: dict[str, str] = {}
    base = {PurePosixPath(f).name: [] for f in cands}
    for f in cands:
        base[PurePosixPath(f).name].append(f)
    for rel in scope:
        t = _parse(root, rel)
        for s, line in (_consts(t) if t else []):
            norm = s.replace("\\", "/")
            if norm.startswith("./"):
                norm = norm[2:]
            if not norm.lower().endswith(DATA_EXT):
                continue
            hit = norm if norm in cands else (base[PurePosixPath(norm).name][0] if len(base.get(PurePosixPath(norm).name, [])) == 1 else None)
            if hit:
                referenced.setdefault(hit, f"{rel}:{line} référence « {s} »")
    pool, how = (sorted(referenced), "référencé dans le code") if referenced else (sorted(cands), "seul fichier de données du projet")
    if len(pool) > 1:
        train_like = [f for f in pool if "train" in PurePosixPath(f).stem.lower()]
        if len(train_like) == 1:
            pool, how = train_like, "unique fichier « train » parmi les candidats"
    if len(pool) != 1:
        return None, f"plusieurs fichiers de données possibles : {pool[:5]}"
    return pool[0], (referenced.get(pool[0]) or how)


def _read_head(root, rel, max_rows=5000, max_mb=50):
    import pandas as pd
    p = root / rel
    if p.stat().st_size > max_mb * 1e6:
        return None, "fichier trop gros pour la vérification"
    try:
        return pd.read_csv(p, sep="\t" if rel.lower().endswith(".tsv") else ",", nrows=max_rows, encoding="utf-8"), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"


def deduce_task(series, scope_trees) -> tuple[str | None, str]:
    from pandas.api.types import is_numeric_dtype
    data_says = "classification" if (not is_numeric_dtype(series) or series.nunique() <= 20) else "regression"
    names = {n.attr for t in scope_trees for n in ast.walk(t) if isinstance(n, ast.Attribute)} | \
            {a.asname or a.name for t in scope_trees for n in ast.walk(t) if isinstance(n, ast.ImportFrom) for a in n.names}
    code_says = {"classification" if CLASSIF_HINT.search(x) else "regression" for x in names if CLASSIF_HINT.search(x) or REGRESS_HINT.search(x)}
    if code_says and code_says != {data_says}:
        return None, f"données : {data_says} ; estimateurs importés : {sorted(code_says)} (conflit)"
    return data_says, f"colonne cible ({series.nunique()} valeurs distinctes)" + (" + estimateurs importés cohérents" if code_says else "")


# ------------------------------------------------------------------ point d'entrée
def deduce_spec(tb, spec: ProjectSpec | None, eps) -> ProjectSpec | None:
    """Complète `spec` (issue du manifeste) avec ce qui est déductible ET vérifié. Ne remplace jamais une valeur du manifeste."""
    root, files = tb.root, sorted(tb.known_files)
    cur = spec.model_copy(deep=True) if spec else ProjectSpec()
    ded: dict[str, str] = dict(cur.deduced)
    train = eps.entries.get("train") if eps else None
    scope = _scope_files(train.file if train else None, files)

    if not cur.data_path:
        path, why = deduce_data_path(root, files, scope)
        if path:
            cur.data_path, ded["data_path"] = path, why
        elif why.startswith("plusieurs"):
            ded["data_path?"] = why          # ambigu : on n'invente pas (le LLM ne tranche pas non plus)
    hints = getattr(tb, "hints", None) or {}
    hspec, hprov = hints.get("spec") or {}, hints.get("provenance") or {}

    def _hint(slot):                         # hypothèse du LLM DÉJÀ confirmée par Python ; trous seulement
        if not getattr(cur, slot) and hspec.get(slot):
            setattr(cur, slot, hspec[slot])
            ded[slot] = hprov.get(slot, "proposé par le LLM ; confirmé par Python")
    _hint("data_path")

    df, head_err = (None, None)
    if cur.data_path and cur.data_path in files and cur.data_path.lower().endswith(DATA_EXT):
        df, head_err = _read_head(root, cur.data_path, max_mb=getattr(tb, "max_data_mb", 50))

    trees = [t for f in scope if (t := _parse(root, f)) is not None]
    if not cur.target and df is not None and len(df.columns) and trees:
        cp = _callee_params(root, files)
        cols = set(map(str, df.columns))
        merged: dict[str, list] = {}
        for t in trees:
            for k, v in target_candidates(t, cp).items():
                merged.setdefault(k, []).extend(v)
        for ctx, found in merged.items():
            ok = list(dict.fromkeys(s for s, _ in found if s in cols))
            if len(ok) == 1:
                line = next(ln for s, ln in found if s == ok[0])
                cur.target, ded["target"] = ok[0], f"{ctx} → « {ok[0]} » (ligne {line}) ; colonne présente dans {cur.data_path}"
                break
            if len(ok) > 1:
                ded["target?"] = f"ambigu ({ctx}) : {ok[:4]}"
                break
        else:
            conv = [c for c in df.columns if str(c).lower() in CONVENTIONAL_COLUMNS]
            if len(conv) == 1:
                cur.target, ded["target"] = str(conv[0]), f"nom de colonne conventionnel « {conv[0]} » présent dans {cur.data_path}"

    _hint("target")
    if cur.target and not cur.task and df is not None and cur.target in df.columns:
        task, why = deduce_task(df[cur.target], trees)
        if task:
            cur.task, ded["task"] = task, why

    _hint("task")
    cur.deduced = ded
    if not any((cur.task, cur.target, cur.primary_metric, cur.data_path)) and not ded:      # les notes d'ambiguïté (« data_path? ») doivent survivre
        return None
    return cur
