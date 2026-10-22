"""Generation deterministe des fichiers a partir du rapport (templates Jinja).

Le LLM n'intervient qu'en correction ou pour les parties que le template ne couvre pas.
"""
from __future__ import annotations

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import config
from .schemas import AnalysisReport

_PY_TYPES = {"int": "int", "float": "float", "str": "str", "bool": "bool"}


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(config.TEMPLATES_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _requirements(report: AnalysisReport) -> list[str]:
    reqs = [f"{p.name}=={p.version}" if p.version else p.name for p in report.dependencies.packages]
    have = {p.name.lower() for p in report.dependencies.packages}
    for extra in report.dependencies.api_packages_to_add:
        if extra.split("[")[0].lower() not in have:
            reqs.append(extra)
    return reqs


def _context(report: AnalysisReport) -> dict:
    ep = report.inference.entrypoint
    gen = report.ml.model_generation
    return {
        "r": report,
        "port": report.existing_app.port or 8000,
        "py_version": report.python.version,
        "py_minor": ".".join(report.python.version.split(".")[:2]),   # 3.11.4 -> 3.11 (dossier site-packages)
        "base_image": f"python:{report.python.version}-slim",
        "requirements": _requirements(report),
        "os_packages": report.dependencies.os_packages,
        "fields": [
            {"name": f.name, "py_type": _PY_TYPES[f.type], "required": f.required}
            for f in report.io.input_schema
        ],
        "loader_fn": (ep.loader_function or "").split("(")[0].strip() or None,   # None = pas de fonction de chargement dans le rapport
        "predict_module": ep.file.removesuffix(".py").replace("/", "."),
        "custom_module": report.ml.custom_classes_module,
        "model_path": report.ml.model_file,
        "train_command": gen.command if gen and gen.required_before_serving else None,
        "train_files": gen.requires_files if gen else [],
        "runtime_files": report.project.runtime_files,
        "supports_proba": report.inference.supports_proba,
        "env_vars": report.existing_app.env_vars,
        "use_compose": bool(report.existing_app.external_services),
        "ignore": report.project.ignorable_paths,
    }


def generate_all(report: AnalysisReport) -> dict[str, str]:
    """Retourne {chemin_relatif: contenu}."""
    env, ctx = _env(), _context(report)
    files = {
        "api.py": env.get_template("api.py.j2").render(**ctx),
        "requirements.api.txt": "\n".join(ctx["requirements"]) + "\n",
        "Dockerfile": env.get_template("Dockerfile.j2").render(**ctx),
        ".dockerignore": env.get_template("dockerignore.j2").render(**ctx),
    }
    if ctx["use_compose"]:
        files["docker-compose.yml"] = env.get_template("docker-compose.yml.j2").render(**ctx)
    return files
