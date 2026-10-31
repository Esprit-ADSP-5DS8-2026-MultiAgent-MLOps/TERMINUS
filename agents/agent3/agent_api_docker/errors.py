"""Taxonomie des erreurs et routage (code API vs Docker)."""
from __future__ import annotations

import re

# (categorie, cible de correction, motif regex sur les logs)
RULES: list[tuple[str, str, str]] = [
    # cible "env" = probleme d'environnement (Docker arrete, absent...) : aucun patch de fichier genere ne peut le corriger
    ("DOCKER_DAEMON", "env", r"Cannot connect to the Docker daemon|error during connect|docker_engine|docker daemon is not running|Is the docker daemon running"),
    ("API_SYNTAX", "api", r"SyntaxError|IndentationError"),
    ("API_MODEL_LOAD", "api", r"NotADirectoryError|IsADirectoryError|Can't get attribute|No such file.*(model|\.joblib|\.pkl|\.pt\b|\.h5|\.onnx)|InconsistentVersionWarning"),
    ("API_IMPORT", "api", r"ImportError|cannot import name"),
    ("DEP_MISSING_PYTHON", "docker", r"ModuleNotFoundError|No matching distribution"),
    ("DEP_VERSION_CONFLICT", "docker", r"ResolutionImpossible|conflicting dependencies"),
    ("DEP_MISSING_SYSTEM", "docker", r"gcc: not found|fatal error: .*\.h"),
    ("BASE_IMAGE", "docker", r"manifest unknown|pull access denied|not found: manifest"),
    ("COPY_PATH", "docker", r"COPY failed|file not found in build context|not found"),
    ("HOST_BINDING", "docker", r"Connection refused|Connection reset"),
    ("PORT_MISMATCH", "docker", r"address already in use"),
    ("PERMISSION", "docker", r"Permission denied"),
    ("API_RUNTIME_500", "api", r"Traceback|Internal Server Error"),
]


def classify(logs: str) -> tuple[str, str]:
    """Retourne (categorie, cible) ; ('UNKNOWN', 'docker') par defaut."""
    for category, target, pattern in RULES:
        if re.search(pattern, logs, re.IGNORECASE):
            return category, target
    return "UNKNOWN", "docker"


def tail_relevant(logs: str, max_lines: int = 100) -> str:
    """Garde les dernieres lignes + celles contenant ERROR, pour ne pas noyer le LLM."""
    lines = logs.splitlines()
    errors = [l for l in lines if re.search(r"error|exception|traceback", l, re.I)]
    return "\n".join(dict.fromkeys(errors[-30:] + lines[-max_lines:]))
