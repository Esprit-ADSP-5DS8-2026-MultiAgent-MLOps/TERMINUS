"""Exécution isolée du code du projet. Deux backends :
 - DockerSandbox     : réseau coupé, FS en lecture seule, utilisateur non-root, CPU/mémoire/PID limités (recommandé).
 - SubprocessSandbox : repli local (copie temporaire, rlimits, environnement épuré, réseau coupé via `unshare` si possible).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


def _sub(value: str, token: str) -> str:
    return value.replace("{W}", token)


class SubprocessSandbox:
    name = "subprocess"

    def __init__(self, memory_mb: int = 4096, python: str | None = None):
        self.memory_mb = memory_mb
        self.python = python or sys.executable   # interpréteur utilisé pour exécuter le projet
        self.network_isolated = self._probe_unshare()

    @staticmethod
    def _probe_unshare() -> bool:
        if not shutil.which("unshare"):
            return False
        try:
            return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
        except Exception:
            return False

    def describe(self) -> dict:
        return {"backend": self.name, "python": self.python, "network_isolated": self.network_isolated, "memory_mb": self.memory_mb,
                "note": "repli local : moins étanche que Docker (pas de FS en lecture seule ni de non-root)"}

    def _limits(self, cpu_s: int):
        mem = self.memory_mb * 1024 * 1024

        def apply():
            if resource is None:
                return
            # macOS refuse RLIMIT_AS (et parfois une limite au-dessus de la limite dure) :
            # une exception ici fait échouer tout le sous-processus, donc on ignore la limite refusée.
            for kind, value in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu_s),
                                (resource.RLIMIT_FSIZE, 1 << 30), (resource.RLIMIT_CORE, 0)):
                try:
                    _, hard = resource.getrlimit(kind)
                    if hard != resource.RLIM_INFINITY:
                        value = min(value, hard)
                    resource.setrlimit(kind, (value, value))
                except (ValueError, OSError):
                    pass
        return apply

    def run(self, argv, work: Path, cwd: Path, env_extra: dict, timeout: int) -> RunResult:
        cmd = [self.python if a == "python" else _sub(a, str(work)) for a in argv]
        if self.network_isolated:
            cmd = ["unshare", "-rn"] + cmd
        (work / "tmp").mkdir(exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(work), "TMPDIR": str(work / "tmp"),
            "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            **{k: _sub(v, str(work)) for k, v in env_extra.items()},
        }
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True, preexec_fn=self._limits(timeout + 10))
        try:
            out, err = proc.communicate(timeout=timeout)
            return RunResult(proc.returncode, out, err, False, time.time() - t0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            return RunResult(None, out or "", err or "", True, time.time() - t0)


def ensure_venv(requirements_file: Path, cache_dir=None, timeout: int = 900):
    """Crée (ou réutilise) un venv isolé contenant EXACTEMENT les dépendances épinglées du projet.

    Retourne (chemin_python, None) ou (None, message_d_erreur). Le venv est mis en cache par hash du
    requirements.txt : les validations suivantes sont instantanées.
    Sécurité : `--only-binary :all:` interdit les sdists, donc aucun setup.py du projet n'est exécuté.
    """
    key = hashlib.sha256(requirements_file.read_bytes() + f"py{sys.version_info.major}.{sys.version_info.minor}".encode()).hexdigest()[:16]
    root = Path(cache_dir or os.environ.get("CV_CACHE_DIR") or Path.home() / ".cache" / "contract-validator")
    base = root / "venvs" / key
    py = base / "bin" / "python"
    if (base / ".ok").exists() and py.exists():
        return py, None
    shutil.rmtree(base, ignore_errors=True)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run([sys.executable, "-m", "venv", str(base)], capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return None, ("Création du venv impossible (sous Debian/Ubuntu : `sudo apt install python3-venv`) : " + (r.stderr or r.stdout)[-300:])
        r = subprocess.run([str(py), "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "--only-binary", ":all:",
                            "-r", str(requirements_file)], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            shutil.rmtree(base, ignore_errors=True)
            return None, (r.stderr or r.stdout).strip()[-600:]
    except subprocess.TimeoutExpired:
        shutil.rmtree(base, ignore_errors=True)
        return None, f"Installation des dépendances > {timeout}s."
    (base / ".ok").write_text("ok")
    return py, None


class DockerSandbox:
    """Non testé dans l'environnement de développement (Docker indisponible) : à valider sur votre machine."""
    name = "docker"
    network_isolated = True

    def __init__(self, image: str, memory_mb: int = 2048):
        self.image, self.memory_mb = image, memory_mb

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False

    @classmethod
    def build_for(cls, project_dir: Path, memory_mb: int, python: str = "3.12"):
        """Construit une image avec EXACTEMENT les dépendances épinglées du projet (le réseau n'est utilisé qu'ici)."""
        req = (project_dir / "requirements.txt").read_bytes()
        tag = "cv-sandbox-" + hashlib.sha256(req + python.encode()).hexdigest()[:12]
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
            with tempfile.TemporaryDirectory() as ctx:
                Path(ctx, "requirements.txt").write_bytes(req)
                Path(ctx, "Dockerfile").write_text(
                    f"FROM python:{python}-slim\nCOPY requirements.txt /tmp/requirements.txt\n"
                    "RUN pip install --no-cache-dir -r /tmp/requirements.txt\n")
                subprocess.run(["docker", "build", "-q", "-t", tag, ctx], check=True, capture_output=True, timeout=900)
        return cls(tag, memory_mb)

    def describe(self) -> dict:
        return {"backend": self.name, "image": self.image, "network_isolated": True, "memory_mb": self.memory_mb}

    def run(self, argv, work: Path, cwd: Path, env_extra: dict, timeout: int) -> RunResult:
        os.system(f"chmod -R a+rwX '{work}'")  # l'utilisateur 65534 doit pouvoir écrire dans /work
        name = "cv-" + uuid.uuid4().hex[:10]
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--cpus", "2", "--memory", f"{self.memory_mb}m", "--pids-limit", "256", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--user", "65534:65534", "-v", f"{work}:/work:rw",
               "-w", str(cwd).replace(str(work), "/work"), "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp"]
        for k, v in env_extra.items():
            cmd += ["-e", f"{k}={_sub(v, '/work')}"]
        cmd += [self.image] + [_sub(a, "/work") for a in argv]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
            return RunResult(r.returncode, r.stdout, r.stderr, False, time.time() - t0)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return RunResult(None, "", "", True, time.time() - t0)


def pick_sandbox(pref: str, project_dir: Path, memory_mb: int):
    """Retourne (sandbox, note). `auto` : Docker si disponible, sinon repli subprocess."""
    note = None
    if pref in ("auto", "docker"):
        if DockerSandbox.available():
            try:
                return DockerSandbox.build_for(project_dir, memory_mb), None
            except Exception as e:
                note = f"Docker disponible mais build impossible ({type(e).__name__}) : repli sur subprocess."
        elif pref == "docker":
            note = "Docker demandé mais indisponible : repli sur subprocess."
    return SubprocessSandbox(memory_mb), note
