"""Remettre un projet à zéro : runs, checkpoints, modèle MLflow, conteneur, monitoring de l'Agent 4.

    python -m orchestrator reset credit_risk            # montre ce qui sera fait
    python -m orchestrator reset credit_risk --yes      # le fait

Rien n'est supprimé sans retour possible, sauf le checkpoint et l'enregistrement MLflow : les dossiers
(runs/<id>, agents/agent4/runs/<projet>) sont déplacés dans une archive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.envscope import read_env_file

from .checkpoint import get_checkpointer
from .graph import Run, RunNotFound
from .runlock import is_busy

REPO = Path(__file__).resolve().parent.parent


def agent4_runs_dir() -> Path:
    return Path(os.environ.get("MONITORING_RUNS_DIR") or REPO / "agents" / "agent4" / "runs").expanduser()


def project_of(state: dict) -> str:
    from .api import project_of as name
    return name(state)


@dataclass
class Plan:
    project: str
    runs: list[str] = field(default_factory=list)
    busy: list[str] = field(default_factory=list)
    model: bool = False
    container: bool = False
    monitoring: Path | None = None

    def empty(self) -> bool:
        return not (self.runs or self.model or self.container or self.monitoring)


def _mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI")
                            or f"sqlite:///{REPO / 'runs' / 'mlflow' / 'mlflow.db'}")
    return MlflowClient()


def _has_model(project: str) -> bool:
    from mlflow.exceptions import MlflowException
    try:
        _mlflow_client().get_registered_model(project)
        return True
    except MlflowException:  # modèle absent du registre
        return False


def _docker(*args: str) -> subprocess.CompletedProcess | None:
    if not shutil.which("docker"):
        return None
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60, check=False)


def _has_container(project: str) -> bool:
    r = _docker("ps", "-a", "--filter", f"name=^mlops-{project}$", "--format", "{{.Names}}")
    return bool(r and r.returncode == 0 and r.stdout.strip())


def plan(project: str, runs_dir: str | Path = "runs", saver=None) -> Plan:
    p = Plan(project)
    root = Path(runs_dir)
    saver = saver or get_checkpointer(runs_dir)
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir():
            continue
        try:
            run = Run.load(d.name, str(runs_dir), saver)
        except RunNotFound:            # mlflow, check... : pas un run
            continue
        if project_of(run.state) == project:
            (p.busy if is_busy(d) else p.runs).append(d.name)
    p.model = _has_model(project)
    p.container = _has_container(project)
    mon = agent4_runs_dir() / project
    p.monitoring = mon if mon.is_dir() else None
    return p


def describe(p: Plan) -> list[str]:
    lines = [f"runs : {', '.join(p.runs) if p.runs else 'aucun'}"]
    if p.busy:
        lines.append(f"runs en cours (laissés en place) : {', '.join(p.busy)}")
    lines.append(f"modèle MLflow « {p.project} » : {'supprimé du registre' if p.model else 'absent'}")
    lines.append(f"conteneur mlops-{p.project} : {'supprimé' if p.container else 'absent'}")
    lines.append(f"monitoring Agent 4 : {p.monitoring if p.monitoring else 'absent'}")
    return lines


def apply(p: Plan, runs_dir: str | Path = "runs", archive: Path | None = None, saver=None) -> Path:
    archive = (archive or Path("~/mlops-archive").expanduser()) / f"{datetime.now():%Y%m%d-%H%M%S}-{p.project}"
    saver = saver or get_checkpointer(runs_dir)
    for run_id in p.runs:
        dest = archive / "runs" / run_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(runs_dir) / run_id), dest)
        saver.delete_thread(run_id)
    if p.model:
        _mlflow_client().delete_registered_model(p.project)
    if p.container:
        _docker("rm", "-f", f"mlops-{p.project}")
    if p.monitoring:
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p.monitoring), archive / "agent4")
    return archive


def repo_context(runs_dir: str) -> str:
    """Même résultat d'où qu'on lance la commande (ex. depuis agents/agent4) : le .env de la racine
    complète l'environnement, et un dossier de runs relatif part de la racine du dépôt."""
    for key, value in read_env_file(REPO / ".env").items():
        os.environ.setdefault(key, value)
    return runs_dir if Path(runs_dir).is_absolute() else str(REPO / runs_dir)


def main(project: str, runs_dir: str = "runs", yes: bool = False) -> int:
    runs_dir = repo_context(runs_dir)
    p = plan(project, runs_dir)
    print(f"Remise à zéro de « {project} » (runs : {runs_dir})")
    for line in describe(p):
        print(f"  {line}")
    if p.empty():
        print("Rien à faire.")
        return 0
    if not yes:
        print("\nRien n'a été modifié. Pour le faire : ajouter --yes "
              "(arrêter d'abord la gateway de l'Agent 4 si elle surveille ce projet).")
        return 0
    archive = apply(p, runs_dir)
    print(f"\nFait. Dossiers archivés dans {archive}")
    return 0
