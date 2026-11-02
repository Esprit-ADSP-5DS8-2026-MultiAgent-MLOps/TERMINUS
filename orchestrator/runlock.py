"""
Un run ne s'exécute qu'à un endroit à la fois (API de Terminus, service Slack, ligne de commande).

    with run_lock(run_dir):
        run.answer(decision)

Le verrou est un fichier runs/<run_id>/.busy qui contient le PID du processus. Un verrou laissé par un
processus mort (plantage, Ctrl+C brutal) est repris automatiquement.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK = ".busy"


class RunBusy(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # processus d'un autre utilisateur : vivant
        return True
    return True


def holder(run_dir: Path) -> int | None:
    """PID du processus qui exécute ce run, ou None."""
    try:
        pid = int((Path(run_dir) / LOCK).read_text().strip() or 0)
    except (OSError, ValueError):
        return None
    return pid if pid and _alive(pid) else None


def is_busy(run_dir: Path) -> bool:
    return holder(run_dir) is not None


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    path = Path(run_dir) / LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if holder(run_dir) is not None:
                raise RunBusy(f"le run {Path(run_dir).name} est déjà en cours d'exécution") from None
            path.unlink(missing_ok=True)          # verrou d'un processus mort
    else:
        raise RunBusy(f"verrou impossible sur {path}")
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
