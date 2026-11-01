"""Noyau commun des trois agents : schémas v2, boucle d'agent générique, client LLM, progression, outils, sandbox, handoff."""
from .schemas import SCHEMA_VERSION  # noqa: F401


def _load_dotenv() -> None:
    """Charge le fichier .env (racine du dépôt, sinon dossier courant) dans os.environ, sans écraser les variables déjà exportées."""
    import os
    from pathlib import Path
    for path in (Path(__file__).resolve().parents[1] / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            if k.strip() and v:
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v
        return


_load_dotenv()