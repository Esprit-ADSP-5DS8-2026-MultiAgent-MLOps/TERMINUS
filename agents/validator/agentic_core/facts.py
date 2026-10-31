"""Faits statiques sur un projet : appels, imports, syntaxe, secrets, chemins. AUCUN jugement, AUCUNE exécution.

Le LLM interprète ces faits ; l'infrastructure s'en sert aussi comme GARDE-FOU (refuser d'exécuter un projet dont
les faits sont manifestement dangereux). Les notebooks (.ipynb) sont lus cellule de code par cellule de code.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from validator.check_static import (ABS_BLOCK, DESTRUCTIVE, SECRET_FILES, SECRET_PATTERNS, SECRET_SCAN_EXT, SHELL_CALLS, URL,
                                    qualname)
from validator.contract import ARTIFACT_EXTENSIONS, NETWORK_MODULES
from validator.fs import read_text_safe, walk_project

from .introspection import SEED_CALL, dist_for_import

MAX_ITEMS = 25
MAX_CALLS = 600                       # appels tiers conservés pour l'introspection (jamais montrés au LLM : clés en « _ »)


def notebook_code(text: str) -> str | None:
    """Concatène les cellules de code d'un notebook ; les lignes `!cmd` deviennent un appel os.system visible, `%magic` est ignoré."""
    try:
        nb = json.loads(text)
    except ValueError:
        return None
    out = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        lines = []
        for line in src.splitlines():
            s = line.lstrip()
            if s.startswith("!"):
                lines.append(f"__import__('os').system({s[1:]!r})  # shell escape notebook")
            elif s.startswith("%"):
                lines.append("pass  # magic")
            else:
                lines.append(line)
        out.append(f"# --- cell {i} ---\n" + "\n".join(lines))
    return "\n".join(out)


def python_sources(root: Path, files) -> dict[str, str]:
    srcs: dict[str, str] = {}
    for rel in files:
        text = None
        if rel.suffix == ".py":
            text = read_text_safe(root / rel)
        elif rel.suffix == ".ipynb":
            raw = read_text_safe(root / rel, limit=5_000_000)
            text = notebook_code(raw) if raw else None
        if text is not None:
            srcs[rel.as_posix()] = text
    return srcs


def _cap(items):
    return items[:MAX_ITEMS] + ([{"note": f"{len(items) - MAX_ITEMS} autres omis"}] if len(items) > MAX_ITEMS else [])


def static_facts(root, files=None) -> dict:
    root = Path(root)
    if files is None:
        files, _ = walk_project(root)
    srcs = python_sources(root, files)
    local = {Path(p).stem for p in srcs} | {part for p in srcs for part in Path(p).parts[:-1]}   # tous les dossiers : src/, src/pkg/…
    stdlib = set(sys.stdlib_module_names)
    f = {k: [] for k in ("syntax_errors", "dangerous_calls", "network_imports", "network_urls", "interactive_input", "absolute_paths",
                         "destructive_fs", "pickle_usage", "secrets", "committed_artifacts", "sensitive_files")}
    third: dict[str, str] = {}
    tp_mods: dict[str, str] = {}        # imports tiers BRUTS (nom d'import) -> « fichier:ligne »
    tp_calls: list[dict] = []           # appels qualifiés vers des paquets tiers (symbole, mots-clés, constantes)
    functions: dict[str, list] = {}
    global_seed = False
    for rel, text in srcs.items():
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as e:
            f["syntax_errors"].append({"file": rel, "line": e.lineno, "msg": e.msg})
            continue
        aliases = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                    mods = [(a.name.split(".")[0], n.lineno)]
                    for m, ln in mods:
                        _imp(f, third, m, ln, rel, local, stdlib, tp_mods)
            elif isinstance(n, ast.ImportFrom) and not n.level:
                mod = n.module or ""
                for a in n.names:
                    aliases[a.asname or a.name] = f"{mod}.{a.name}"
                _imp(f, third, mod.split(".")[0], n.lineno, rel, local, stdlib, tp_mods)
        tp_roots = {v.split(".")[0] for v in aliases.values()} - stdlib - local      # racines d'imports tiers de CE fichier
        functions[rel] = [{"name": n.name, "params": [a.arg for a in n.args.posonlyargs + n.args.args], "line": n.lineno,
                           "required": len(n.args.posonlyargs + n.args.args) - len(n.args.defaults)}
                          for n in tree.body if isinstance(n, ast.FunctionDef)]
        doc = {id(b[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
               for b in [n.body] if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc:
                v = n.value
                if URL.search(v):
                    f["network_urls"].append({"file": rel, "line": n.lineno, "value": v[:80]})
                elif not any(c.isspace() for c in v) and ABS_BLOCK.match(v):
                    f["absolute_paths"].append({"file": rel, "line": n.lineno, "value": v[:80]})
            if not isinstance(n, ast.Call):
                continue
            q = qualname(n.func, aliases) or ""
            kws = {k.arg for k in n.keywords}
            last = q.split(".")[-1]
            if "." in q and q.split(".")[0] in tp_roots and len(tp_calls) < MAX_CALLS:
                consts = {k.arg: k.value.value for k in n.keywords
                          if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, (bool, int, float, str, type(None)))
                          and len(repr(k.value.value)) < 30}
                tp_calls.append({"file": rel, "line": n.lineno, "symbol": q, "kws": sorted(k for k in kws if k), "starkw": None in kws,
                                 "nargs": len(n.args), "star": any(isinstance(a, ast.Starred) for a in n.args), "consts": consts})
            if q in ("eval", "exec", "__import__", "compile") or q.startswith(SHELL_CALLS):
                if q != "compile":
                    f["dangerous_calls"].append({"file": rel, "line": n.lineno, "call": q})
            elif q == "getattr" and n.args and isinstance(n.args[0], ast.Name) and aliases.get(n.args[0].id, n.args[0].id) in ("os", "subprocess", "sys"):
                f["dangerous_calls"].append({"file": rel, "line": n.lineno, "call": "getattr(<module système>, …) : appel dissimulé possible"})
            elif q == "input" or q.startswith("sys.stdin"):
                f["interactive_input"].append({"file": rel, "line": n.lineno, "call": q})
            elif q in DESTRUCTIVE:
                f["destructive_fs"].append({"file": rel, "line": n.lineno, "call": q})
            elif q.startswith("pickle."):
                f["pickle_usage"].append({"file": rel, "line": n.lineno, "call": q})
            if SEED_CALL.search(last):                 # graine globale (random.seed, np.random.seed, torch.manual_seed, set_seed…)
                global_seed = True
    for rel in files:
        sp = rel.as_posix()
        if SECRET_FILES.search(sp):
            f["sensitive_files"].append(sp)
        if rel.suffix.lower() in ARTIFACT_EXTENSIONS:
            f["committed_artifacts"].append(sp)
        if rel.suffix.lower() in SECRET_SCAN_EXT or rel.name == ".env":
            for i, line in enumerate((read_text_safe(root / rel) or "").splitlines(), 1):
                for label, rx in SECRET_PATTERNS:
                    m = rx.search(line)
                    if m:
                        f["secrets"].append({"file": sp, "line": i, "kind": label, "masked": m.group(0)[:4] + "…[masqué]"})
                        break
    out = {k: _cap(v) if isinstance(v, list) else v for k, v in f.items() if v}
    out["python_files_analyzed"] = sorted(srcs)
    out["_third_party_modules"] = dict(sorted(tp_mods.items()))     # entrées de l'introspection du venv (préfixe « _ » : non montrées au LLM)
    out["_third_party_calls"] = tp_calls
    out["_global_seed"] = global_seed
    out["third_party_imports"] = {k: v for k, v in sorted(third.items())}
    out["functions"] = {k: v for k, v in functions.items() if v}
    out["notebooks"] = sorted(r.as_posix() for r in files if r.suffix == ".ipynb")
    return out


def _imp(f, third, mod, line, rel, local, stdlib, tp_mods):
    if not mod:
        return
    if mod in NETWORK_MODULES:
        f["network_imports"].append({"file": rel, "line": line, "module": mod})
    if mod in stdlib or mod in local:
        return
    tp_mods.setdefault(mod, f"{rel}:{line}")
    third.setdefault(dist_for_import(mod), f"{rel}:{line}")   # estimation par métadonnées ; l'introspection du venv fait foi


UNSAFE_KEYS = ("dangerous_calls", "network_imports", "network_urls", "secrets", "sensitive_files", "syntax_errors")


def unsafe_to_execute(facts: dict) -> list[str]:
    """Garde-fou d'infrastructure : raisons de NE PAS exécuter le projet, même si le LLM le demande."""
    return [f"{k} ({len(facts[k])})" for k in UNSAFE_KEYS if facts.get(k)]
