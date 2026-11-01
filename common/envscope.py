"""
Le .env d'un agent, appliqué seulement pendant que cet agent travaille.

Tous les agents tournent dans le processus de l'orchestrateur. Certains chargent leur .env dans os.environ
(GROQ_API_KEY...) : sans précaution, la clé d'un agent devient celle de tous les autres pour le reste du run.

    with agent_env(AGENT_DIR / ".env"):
        resultat = agent(...)

Pendant le bloc : les valeurs non vides du fichier remplacent celles du processus (l'agent utilise SES clés).
Après le bloc : os.environ redevient exactement ce qu'il était, y compris ce que l'agent y a ajouté lui-même.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE (commentaires, lignes vides et « export » acceptés ; guillemets retirés ; valeurs vides ignorées)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            values[key] = value
    return values


@contextmanager
def agent_env(env_file: str | Path) -> Iterator[dict[str, str]]:
    saved = dict(os.environ)
    values = read_env_file(Path(env_file))
    os.environ.update(values)
    try:
        yield values
    finally:
        os.environ.clear()
        os.environ.update(saved)
