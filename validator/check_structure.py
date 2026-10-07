"""Niveau 1 - Structure : fichiers requis (C1), limites (C12), dépendances (C2/C6)."""
from __future__ import annotations

from packaging.requirements import InvalidRequirement, Requirement

from .contract import norm_dist
from .fs import is_inside, read_text_safe
from .models import Severity as S


def parse_requirements(text: str):
    """Retourne une liste de (lineno, brut, Requirement|None, erreur|None)."""
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "://" in line or line.startswith("git+"):
            out.append((i, line, None, "option"))
            continue
        try:
            out.append((i, line, Requirement(line), None))
        except InvalidRequirement as e:
            out.append((i, line, None, str(e)))
    return out


def _is_pinned(req: Requirement) -> bool:
    specs = list(req.specifier)
    return len(specs) == 1 and specs[0].operator in ("==", "===") and not specs[0].version.endswith(".*")


def run(ctx):
    cfg = ctx.cfg
    if not ctx.root.is_dir():
        ctx.add("C1", "C1.NOT_A_DIRECTORY", S.BLOCKER, f"Le chemin {ctx.root} n'est pas un dossier.")
        return

    # --- C12 : liens symboliques, volume ---------------------------------------------------
    for link in ctx.symlinks:
        if not is_inside(ctx.root / link, ctx.root):
            ctx.add("C12", "C12.SYMLINK_ESCAPE", S.BLOCKER, "Lien symbolique pointant hors du projet.",
                    file=link, hint="Supprimez le lien : le projet doit être autonome.")
    total = sum((ctx.root / f).stat().st_size for f in ctx.files)
    ctx.facts["n_files"], ctx.facts["size_mb"] = len(ctx.files), round(total / 1e6, 2)
    if len(ctx.files) > cfg.max_files:
        ctx.add("C12", "C12.TOO_MANY_FILES", S.BLOCKER, f"{len(ctx.files)} fichiers (max {cfg.max_files}).")
    if total > cfg.max_project_mb * 1e6:
        ctx.add("C12", "C12.PROJECT_TOO_LARGE", S.BLOCKER, f"Projet de {total/1e6:.0f} Mo (max {cfg.max_project_mb}).")

    # --- C1 : fichiers requis à la racine ----------------------------------------------------
    present = {str(f) for f in ctx.files}
    for name in cfg.required_files:
        if name not in present:
            where = [str(f) for f in ctx.files if f.name == name]
            hint = f"Le fichier existe ailleurs ({where[0]}) : déplacez-le à la racine." if where else \
                   f"Ajoutez {name} à la racine du projet."
            ctx.add("C1", "C1.MISSING_FILE", S.BLOCKER, f"Fichier requis absent à la racine : {name}", file=name, hint=hint)

    # --- C6 / C2 : requirements.txt -----------------------------------------------------------
    req_path = ctx.root / "requirements.txt"
    reqs = {}
    if req_path.exists():
        for lineno, raw, req, err in parse_requirements(read_text_safe(req_path) or ""):
            if err == "option":
                ctx.add("C6", "C6.FORBIDDEN_LINE", S.BLOCKER, "Option, URL ou dépôt Git interdits dans requirements.txt.",
                        file="requirements.txt", line=lineno, evidence=raw, hint="Utilisez uniquement des paquets PyPI épinglés (nom==version).")
                continue
            if req is None:
                ctx.add("C6", "C6.INVALID_LINE", S.BLOCKER, f"Ligne illisible : {err}", file="requirements.txt", line=lineno, evidence=raw)
                continue
            name = norm_dist(req.name)
            if _is_pinned(req):
                reqs[name] = next(iter(req.specifier)).version
            else:
                reqs[name] = None
                ctx.add("C6", "C6.UNPINNED", S.BLOCKER, f"Dépendance non épinglée : {raw}", file="requirements.txt", line=lineno,
                        evidence=raw, hint=f"Épinglez la version exacte, ex. `{req.name}==<version testée>` (pip freeze).")
        ctx.facts["requirements"] = reqs
        if "scikit-learn" not in reqs:
            ctx.add("C2", "C2.NO_SKLEARN", S.BLOCKER, "scikit-learn absent de requirements.txt (périmètre V1).", file="requirements.txt",
                    hint="Le contrat V1 ne couvre que les modèles scikit-learn.")
        if "pandas" not in reqs:
            ctx.add("C2", "C2.NO_PANDAS", S.BLOCKER, "pandas absent de requirements.txt (predict accepte un DataFrame).", file="requirements.txt")
        for dist in sorted(set(reqs) & cfg.blocked_frameworks):
            ctx.add("C2", "C2.UNSUPPORTED_FRAMEWORK", S.BLOCKER, f"Framework hors périmètre V1 : {dist}", file="requirements.txt",
                    hint=f"Retirez {dist} ou élargissez le contrat (--allow-framework {dist}).")
