"""Exécuté DANS le sandbox, avec l'interpréteur du venv du projet. Introspection des paquets INSTALLÉS : jamais le code du candidat.

Entrée  (--request) : {"modules": ["sklearn", "imblearn"], "symbols": ["sklearn.ensemble.RandomForestClassifier", ...]}
Sortie  (--report)  : {"introspected": true, "modules": {module: [distributions]}, "symbols": {symbole: description}, "errors": {...}}
Ce script est lancé par son chemin : le dossier du projet n'est PAS sur sys.path, donc un module local ne peut pas masquer un paquet.
"""
import argparse
import importlib
import inspect
import json
import sys

POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
NAMED = POSITIONAL + (inspect.Parameter.KEYWORD_ONLY,)


def distributions(modules):
    try:
        from importlib.metadata import packages_distributions
        table = packages_distributions()
    except Exception:
        table = {}
    return {m: sorted(set(table.get(m, []))) for m in modules}


def resolve(qualified):
    parts = qualified.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(qualified)


def describe(obj):
    kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "other"
    d = {"kind": kind, "has_fit": bool(inspect.isclass(obj) and callable(getattr(obj, "fit", None)))}
    if kind == "other":
        return d
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return d
    ps = list(sig.parameters.values())
    d["params"] = [p.name for p in ps if p.kind in NAMED]
    d["positional"] = [p.name for p in ps if p.kind in POSITIONAL]
    d["var_kw"] = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps)
    d["defaults"] = {p.name: repr(p.default)[:40] for p in ps if p.default is not inspect.Parameter.empty and p.kind in NAMED}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request"), ap.add_argument("--report")
    a = ap.parse_args()
    req = json.load(open(a.request))
    rep = {"introspected": True, "python": sys.version.split()[0], "modules": distributions(req.get("modules") or []), "symbols": {}, "errors": {}}
    for q in (req.get("symbols") or [])[:400]:
        try:
            rep["symbols"][q] = describe(resolve(q))
        except BaseException as e:                       # SystemExit / import à effets de bord compris : on note et on continue
            rep["errors"][q] = f"{type(e).__name__}: {str(e)[:100]}"
    json.dump(rep, open(a.report, "w"), default=str)


if __name__ == "__main__":
    main()
