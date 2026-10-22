"""
Persistance de l'état du pipeline.

LangGraph enregistre l'état après chaque étape (un « checkpoint »). Ces
checkpoints sont écrits dans une base de données : un run en pause ou
interrompu par un plantage peut donc reprendre plus tard, même depuis un
autre terminal.

Deux bases possibles, choisies par la variable d'environnement ORCH_DATABASE_URL :

    (non définie)                              -> SQLite, fichier runs/checkpoints.sqlite
    postgresql://user:mdp@localhost:5432/base  -> PostgreSQL (base prévue par l'architecture)

Le reste de l'orchestrateur ne sait pas quelle base est utilisée.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit

DB_NAME = "checkpoints.sqlite"
ENV_VAR = "ORCH_DATABASE_URL"


def database_url() -> str | None:
    url = os.environ.get(ENV_VAR, "").strip()
    return url or None


def uses_postgres() -> bool:
    url = database_url()
    return bool(url) and url.startswith(("postgresql://", "postgres://"))


def describe_backend(runs_dir: str | Path = "runs") -> str:
    """Description lisible de la base utilisée (sans mot de passe)."""
    if uses_postgres():
        u = urlsplit(database_url())
        return f"PostgreSQL · {u.hostname}:{u.port or 5432}{u.path}"
    return f"SQLite · {Path(runs_dir).resolve() / DB_NAME}"


def get_checkpointer(runs_dir: str | Path = "runs"):
    """Le checkpointer à utiliser : PostgreSQL si configuré, sinon SQLite."""
    if uses_postgres():
        return postgres_checkpointer(database_url())
    return sqlite_checkpointer(runs_dir)


def sqlite_checkpointer(runs_dir: str | Path = "runs"):
    from langgraph.checkpoint.sqlite import SqliteSaver

    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False : Streamlit peut appeler depuis plusieurs threads
    conn = sqlite3.connect(runs_dir / DB_NAME, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def postgres_checkpointer(url: str):
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row
    except ImportError as exc:  # message clair si les paquets manquent
        raise RuntimeError(
            "PostgreSQL demandé mais paquets absents : "
            'pip install "langgraph-checkpoint-postgres==3.2.0" "psycopg[binary]"'
        ) from exc

    try:
        conn = Connection.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    except Exception as exc:
        raise RuntimeError(
            f"connexion PostgreSQL impossible ({describe_backend()}) : {exc}. "
            "La base est-elle démarrée ? (docker compose -f infra/docker-compose.yml up -d)"
        ) from exc
    saver = PostgresSaver(conn)
    saver.setup()  # crée les tables la première fois
    return saver
