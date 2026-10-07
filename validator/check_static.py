"""Niveau 3 - Analyse statique (AST, regex, Bandit optionnel). AUCUN code du projet n'est exécuté."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys

from .contract import (ARTIFACT_EXTENSIONS, IMPORT_TO_DIST, NETWORK_MODULES, SHUFFLE_SPLITTERS,
                       STOCHASTIC_CALLS, norm_dist)
from .fs import read_text_safe
from .models import Severity as S

ABS_BLOCK = re.compile(r"^(?:/(?:home|Users|tmp|var|mnt|opt|root|data|etc|srv|media|workspace|content|kaggle)\b|[A-Za-z]:[\\/])")
ABS_ANY = re.compile(r"^/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+/?$")
URL = re.compile(r"https?://\S+")

SECRET_PATTERNS = [
    ("clé AWS", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("token GitHub", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("clé API (sk-...)", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("clé privée", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("secret en clair", re.compile(r"""(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|token)\b\s*[:=]\s*['"][^'"\s]{8,}['"]""")),
]
SECRET_SCAN_EXT = {".py", ".txt", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml", ".env", ".sh"}
SECRET_FILES = re.compile(r"(^|/)(\.env|id_rsa|id_ed25519|.*\.pem|.*\.key)$")

SHELL_CALLS = ("subprocess.", "os.system", "os.popen", "os.exec", "os.spawn")
DESTRUCTIVE = {"shutil.rmtree", "os.remove", "os.unlink", "os.rmdir"}


def qualname(node, aliases):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(aliases.get(node.id, node.id))
        return ".".join(reversed(parts))
    return None


def _docstring_ids(tree):
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                ids.add(id(b[0].value))
    return ids


def _positional(fn: ast.FunctionDef):
    args = fn.args.posonlyargs + fn.args.args
    return [a.arg for a in args], len(args) - len(fn.args.defaults)


def run(ctx):
    cfg = ctx.cfg
    declared = set((ctx.facts.get("requirements") or {}).keys())
    local_mods = {p.stem for p in ctx.py_files()} | {p.parts[0] for p in ctx.py_files() if len(p.parts) > 1}
    local_mods |= {f[:-3] for f in cfg.required_files if f.endswith(".py")}  # un fichier requis manquant n'est pas un paquet tiers
    stdlib = set(sys.stdlib_module_names)
    trees, global_seed, seed_candidates = {}, False, []
    imported_third_party = set()

    for rel in ctx.py_files():
        text = read_text_safe(ctx.root / rel)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=str(rel))
        except SyntaxError as e:
            ctx.add("C2", "C2.SYNTAX_ERROR", S.BLOCKER, f"Erreur de syntaxe : {e.msg}", file=rel, line=e.lineno)
            continue
        trees[str(rel)] = tree
        aliases, imports = {}, []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                    imports.append((a.name.split(".")[0], n.lineno))
            elif isinstance(n, ast.ImportFrom) and not n.level:
                mod = n.module or ""
                for a in n.names:
                    aliases[a.asname or a.name] = f"{mod}.{a.name}"
                imports.append((mod.split(".")[0], n.lineno))

        # --- imports : réseau, frameworks, non déclarés -------------------------------------
        for root_mod, line in imports:
            if not root_mod:
                continue
            if root_mod in NETWORK_MODULES:
                ctx.add("C11", "C11.NETWORK_IMPORT", S.BLOCKER, f"Import réseau interdit : {root_mod}", file=rel, line=line,
                        hint="Le projet doit fonctionner sans accès réseau (données locales uniquement).")
            if root_mod in stdlib or root_mod in local_mods:
                continue
            dist = norm_dist(IMPORT_TO_DIST.get(root_mod, root_mod))
            imported_third_party.add(dist)
            if dist in cfg.blocked_frameworks:
                ctx.add("C2", "C2.UNSUPPORTED_FRAMEWORK", S.BLOCKER, f"Framework hors périmètre V1 : {root_mod}", file=rel, line=line)
            elif declared and dist not in declared:
                ctx.add("C6", "C6.UNDECLARED_IMPORT", S.WARNING, f"`{root_mod}` importé mais absent de requirements.txt", file=rel, line=line,
                        hint=f"Ajoutez {dist}==<version> à requirements.txt.")

        doc_ids = _docstring_ids(tree)
        has_guard = False
        for n in ast.walk(tree):
            # --- constantes : chemins absolus, URL ------------------------------------------------
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids:
                v = n.value
                if URL.search(v):
                    ctx.add("C11", "C11.NETWORK_URL", S.BLOCKER, "URL dans le code : accès réseau interdit.", file=rel, line=n.lineno, evidence=v[:80])
                elif not any(c.isspace() for c in v) and ABS_BLOCK.match(v):
                    ctx.add("C10", "C10.ABSOLUTE_PATH", S.BLOCKER, "Chemin absolu codé en dur.", file=rel, line=n.lineno, evidence=v[:80],
                            hint="Utilisez des chemins relatifs, ou les paramètres data_path/output_dir.")
                elif ABS_ANY.match(v):
                    ctx.add("C10", "C10.ABSOLUTE_PATH_SUSPECT", S.WARNING, "Chaîne ressemblant à un chemin absolu.", file=rel, line=n.lineno, evidence=v[:80])
            if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__":
                has_guard = True
            if not isinstance(n, ast.Call):
                continue
            q = qualname(n.func, aliases) or ""
            kws = {k.arg for k in n.keywords}
            if q in ("eval", "exec", "__import__"):
                ctx.add("C11", "C11.DANGEROUS_CALL", S.BLOCKER, f"Appel dangereux : {q}()", file=rel, line=n.lineno,
                        hint="eval/exec exécutent du code arbitraire ; remplacez par une logique explicite.")
            elif q.startswith(SHELL_CALLS):
                ctx.add("C11", "C11.DANGEROUS_CALL", S.BLOCKER, f"Exécution de commande système : {q}()", file=rel, line=n.lineno)
            elif q == "input" or q.startswith("sys.stdin"):
                ctx.add("C5", "C5.INTERACTIVE_INPUT", S.BLOCKER, f"Lecture interactive ({q}) : le projet doit tourner sans intervention.", file=rel, line=n.lineno,
                        hint="Remplacez par un paramètre, une variable d'environnement ou une valeur par défaut.")
            elif q in DESTRUCTIVE:
                ctx.add("C10", "C10.DESTRUCTIVE_FS", S.WARNING, f"Opération destructive sur le disque : {q}()", file=rel, line=n.lineno)
            elif q.startswith("pickle."):
                ctx.add("C11", "C11.PICKLE_USAGE", S.WARNING, f"Usage de pickle ({q}) : risque d'exécution de code au chargement.", file=rel, line=n.lineno)
            last = q.split(".")[-1]
            if q.endswith("random.seed") or q in ("random.seed", "numpy.random.seed", "np.random.seed", "set_seed") or last == "seed":
                global_seed = True
            elif last in STOCHASTIC_CALLS and "random_state" not in kws and None not in kws:
                seed_candidates.append((rel, n.lineno, last))
            elif last in SHUFFLE_SPLITTERS and "random_state" not in kws and None not in kws:
                if any(k.arg == "shuffle" and isinstance(k.value, ast.Constant) and k.value.value is True for k in n.keywords):
                    seed_candidates.append((rel, n.lineno, last))
        if rel.name == "main.py" and len(rel.parts) == 1 and not has_guard:
            ctx.add("C5", "C5.NO_MAIN_GUARD", S.WARNING, "main.py sans `if __name__ == \"__main__\":`.", file=rel)

    # --- C9 : graines -----------------------------------------------------------------------------
    if not global_seed:
        for rel, line, name in seed_candidates:
            ctx.add("C9", "C9.MISSING_SEED", S.WARNING, f"{name}() sans random_state : résultats non reproductibles.", file=rel, line=line,
                    hint="Ajoutez random_state=<entier fixe> (constante partagée).")

    # --- C3 / C4 : signatures ----------------------------------------------------------------------
    def check_funcs(fname, specs, rule):
        tree = trees.get(fname)
        if tree is None:
            return
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        reexported = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
        for func, (names, nreq) in specs.items():
            if func in defs:
                got, required = _positional(defs[func])
                if len(got) < len(names) or required > len(names):
                    ctx.add(rule, f"{rule}.BAD_SIGNATURE", S.BLOCKER, f"{func}({', '.join(got)}) : attendu {func}({', '.join(names)}).",
                            file=fname, line=defs[func].lineno, hint=f"Adaptez la signature en {func}({', '.join(names)}).")
                elif got[:len(names)] != names:
                    ctx.add(rule, f"{rule}.PARAM_NAMES", S.WARNING, f"Noms de paramètres différents : {got[:len(names)]} (attendu {names}).",
                            file=fname, line=defs[func].lineno)
            elif func in reexported:
                ctx.add(rule, f"{rule}.REEXPORTED", S.INFO, f"{func} ré-exporté depuis un autre module : signature vérifiée à l'exécution.", file=fname)
            else:
                ctx.add(rule, f"{rule}.MISSING_FUNCTION", S.BLOCKER, f"Fonction `{func}` absente de {fname}.", file=fname,
                        hint=f"Définissez {func}({', '.join(names)}).")

    check_funcs("train.py", {"train": (["data_path", "output_dir"], 2)}, "C3")
    check_funcs("predict.py", {"load_model": (["model_path"], 1), "predict": (["model", "data"], 2)}, "C4")

    # --- C11 : secrets, artefacts committés ---------------------------------------------------------
    for rel in ctx.files:
        sp = rel.as_posix()
        if SECRET_FILES.search(sp):
            ctx.add("C11", "C11.SECRET_FILE", S.BLOCKER, "Fichier sensible dans le projet (clé/identifiants).", file=rel, hint="Supprimez-le et révoquez le secret.")
        if rel.suffix.lower() in ARTIFACT_EXTENSIONS:
            ctx.add("C11", "C11.COMMITTED_ARTIFACT", S.WARNING, "Artefact binaire commité : ignoré (jamais chargé), le modèle sera ré-entraîné.", file=rel,
                    hint="Retirez-le du dépôt (.gitignore) ; charger un pickle externe peut exécuter du code.")
        if rel.suffix.lower() in SECRET_SCAN_EXT or rel.name == ".env":
            text = read_text_safe(ctx.root / rel)
            for i, line in enumerate((text or "").splitlines(), 1):
                for label, rx in SECRET_PATTERNS:
                    m = rx.search(line)
                    if m:
                        ctx.add("C11", "C11.SECRET", S.BLOCKER, f"Secret probable ({label}) dans le code.", file=rel, line=i,
                                evidence=m.group(0)[:4] + "…[masqué]", hint="Passez-le par variable d'environnement et révoquez-le.")
                        break

    # --- Bandit (optionnel, HIGH uniquement, dédupliqué) ------------------------------------------------
    if ctx.options.get("bandit", True) and ctx.py_files():
        try:
            r = subprocess.run([sys.executable, "-m", "bandit", "-r", str(ctx.root), "-f", "json", "-q", "-lll"],
                               capture_output=True, text=True, timeout=60)
            data = json.loads(r.stdout or "{}")
            known = {(f.file, f.line) for f in ctx.findings if f.rule == "C11"}
            for item in data.get("results", []):
                rel = str(__import__("pathlib").Path(item["filename"]).resolve().relative_to(ctx.root.resolve()))
                if (rel, item["line_number"]) not in known:
                    ctx.add("C11", "C11.BANDIT_HIGH", S.WARNING, f"Bandit : {item['issue_text']}", file=rel, line=item["line_number"], evidence=item["test_id"])
            ctx.facts["bandit"] = "ok"
        except Exception as e:  # bandit absent ou en échec : non bloquant
            ctx.facts["bandit"] = f"indisponible ({type(e).__name__})"
