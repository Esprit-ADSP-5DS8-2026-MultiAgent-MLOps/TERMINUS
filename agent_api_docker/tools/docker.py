"""Outils Docker via la CLI `docker` (aucune dependance Python en plus).

Tout passe par DockerCLI._exec : un seul point d'appel au systeme, facile a remplacer par un faux en test.
Chaque outil retourne un dict {exit_code, logs, ...}. Aucun outil n'interprete les erreurs : c'est le role
de errors.classify (puis du fixer pour les fichiers generes).
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

_SECRET_MARKERS = ("LLM_", "ANTHROPIC_", "OPENAI_", "MISTRAL_", "AWS_")
_SECRET_SUFFIXES = ("_API_KEY", "_SECRET", "_PASSWORD")


class DockerError(Exception):
    """Echec d'une commande Docker qui ne vient pas du code genere."""


class DockerUnavailable(DockerError):
    """Docker absent ou daemon arrete : rien a corriger dans les fichiers generes."""


def _docker_env() -> dict[str, str]:
    """Environnement herite (PATH, DOCKER_HOST, DOCKER_CONFIG...) sans les cles de LLM."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(_SECRET_MARKERS) and not k.endswith(_SECRET_SUFFIXES)}


class DockerCLI:
    def __init__(self, binary: str = "docker"):
        self.binary = binary

    # ------------------------------------------------------------------ bas niveau
    def _exec(self, args: list[str], timeout: int = 60, stdin: Optional[str] = None,
              on_line: Optional[Callable[[str], None]] = None) -> tuple[int, str]:
        """Execute `docker <args>`. Retourne (code_retour, sortie stdout+stderr). 124 = timeout."""
        try:
            proc = subprocess.Popen([self.binary, *args], stdin=subprocess.PIPE if stdin is not None else None,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", env=_docker_env())
        except FileNotFoundError as e:
            raise DockerUnavailable("La commande `docker` est introuvable : installer Docker Desktop / Docker Engine "
                                    "et verifier qu'elle est dans le PATH.") from e
        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout, _kill)
        timer.start()
        lines: list[str] = []
        try:
            if stdin is not None and proc.stdin:
                proc.stdin.write(stdin)
                proc.stdin.close()
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                lines.append(line)
                if on_line:
                    on_line(line)
            proc.wait()
        finally:
            timer.cancel()
        out = "\n".join(lines)
        if timed_out.is_set():
            return 124, out + f"\n[timeout : commande interrompue apres {timeout} s]"
        return proc.returncode, out

    # ------------------------------------------------------------------ disponibilite
    def available(self) -> tuple[bool, str]:
        """(True, version du serveur) si le daemon repond, sinon (False, explication lisible)."""
        try:
            rc, out = self._exec(["version", "--format", "{{.Server.Version}}"], timeout=20)
        except DockerUnavailable as e:
            return False, str(e)
        if rc != 0:
            return False, ("Le daemon Docker ne repond pas : lancer Docker Desktop (ou `sudo systemctl start docker`) "
                           "puis relancer.\n" + out.strip()[-400:])
        return True, out.strip()

    # ------------------------------------------------------------------ build / run
    def build(self, context_dir: str, tag: str, timeout_s: int = 600, dockerfile: str = "Dockerfile",
              on_line: Optional[Callable[[str], None]] = None) -> dict:
        ctx = Path(context_dir).resolve()
        # `-f` est relatif au dossier d'ou l'on LANCE la commande, pas au contexte : chemin absolu obligatoire.
        rc, logs = self._exec(["build", "--progress=plain", "-t", tag, "-f", str(ctx / dockerfile), str(ctx)],
                              timeout=timeout_s, on_line=on_line)
        return {"exit_code": rc, "logs": logs}

    def run(self, image: str, port: int, env: Optional[dict] = None, name: Optional[str] = None) -> dict:
        """Lance le conteneur en arriere-plan, port publie sur 127.0.0.1 uniquement (port hote libre choisi par Docker)."""
        args = ["run", "-d", "-p", f"127.0.0.1::{port}"]
        if name:
            args += ["--name", name]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        rc, out = self._exec(args + [image], timeout=120)
        if rc != 0:
            return {"exit_code": rc, "logs": out, "container_id": None, "host_port": None}
        container_id = out.strip().splitlines()[-1].strip()
        rc2, out2 = self._exec(["port", container_id, f"{port}/tcp"], timeout=20)
        m = re.search(r":(\d+)\s*$", out2.strip().splitlines()[0]) if rc2 == 0 and out2.strip() else None
        if not m:
            return {"exit_code": 1, "logs": f"port {port} non publie : {out2}", "container_id": container_id,
                    "host_port": None}
        return {"exit_code": 0, "logs": out, "container_id": container_id, "host_port": int(m.group(1))}

    def logs(self, container_id: str, tail: int = 200) -> str:
        return self._exec(["logs", "--tail", str(tail), container_id], timeout=30)[1]

    def state(self, container_id: str) -> dict:
        """{status: running|exited|..., exit_code, health: healthy|unhealthy|starting|''}"""
        fmt = "{{.State.Status}}|{{.State.ExitCode}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}"
        rc, out = self._exec(["inspect", "-f", fmt, container_id], timeout=20)
        if rc != 0:
            return {"status": "unknown", "exit_code": None, "health": ""}
        status, code, health = (out.strip().split("|") + ["", "", ""])[:3]
        return {"status": status, "exit_code": int(code) if code.lstrip("-").isdigit() else None, "health": health}

    def cleanup(self, container_id: str) -> None:
        self._exec(["rm", "-f", container_id], timeout=60)

    # ------------------------------------------------------------------ image
    def image_id(self, ref: str) -> str:
        """Identifiant (sha256:...) du contenu de l'image locale ; sert de 'digest' pour l'approbation du push."""
        rc, out = self._exec(["image", "inspect", "-f", "{{.Id}}", ref], timeout=20)
        if rc != 0:
            raise DockerError(f"image introuvable : {ref}\n{out[-300:]}")
        return out.strip()

    def image_size_mb(self, ref: str) -> float:
        rc, out = self._exec(["image", "inspect", "-f", "{{.Size}}", ref], timeout=20)
        return round(int(out.strip()) / 1e6, 1) if rc == 0 and out.strip().isdigit() else 0.0

    def tag(self, source: str, target: str) -> None:
        rc, out = self._exec(["tag", source, target], timeout=20)
        if rc != 0:
            raise DockerError(f"docker tag {source} {target} a echoue : {out[-300:]}")

    def remove_image(self, ref: str) -> None:
        self._exec(["rmi", ref], timeout=60)

    # ------------------------------------------------------------------ Docker Hub
    def login(self, username: str, token: str) -> dict:
        """Le token passe par stdin (--password-stdin) : jamais dans la ligne de commande ni dans les logs."""
        rc, out = self._exec(["login", "-u", username, "--password-stdin"], timeout=60, stdin=token + "\n")
        return {"exit_code": rc, "logs": out.replace(token, "***") if token else out}

    def push(self, ref: str, timeout_s: int = 900, on_line: Optional[Callable[[str], None]] = None) -> dict:
        rc, out = self._exec(["push", ref], timeout=timeout_s, on_line=on_line)
        m = re.search(r"digest:\s*(sha256:[0-9a-f]{64})", out)
        return {"exit_code": rc, "logs": out, "digest": m.group(1) if m else None}

    def logout(self) -> None:
        self._exec(["logout"], timeout=20)


# --------------------------------------------------------------------------- API fonctionnelle (noms d'origine)
_default = DockerCLI()


def docker_build(context_dir: str, tag: str, timeout_s: int = 600) -> dict:
    return _default.build(context_dir, tag, timeout_s)


def docker_run(image: str, port: int, env: dict | None = None) -> dict:
    return _default.run(image, port, env)


def docker_logs(container_id: str) -> str:
    return _default.logs(container_id)


def docker_cleanup(container_id: str) -> None:
    _default.cleanup(container_id)
