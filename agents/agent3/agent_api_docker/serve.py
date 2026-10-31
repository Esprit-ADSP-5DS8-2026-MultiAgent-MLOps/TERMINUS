"""Lance le conteneur de l'image validee et le supprime quand on quitte (Ctrl+C).

    python -m agent_api_docker <projet> --serve-docker   # pipeline complet, puis lancement du conteneur
    python -m agent_api_docker <projet> --run-only       # relance la derniere image validee, sans rien regenerer

Garanties : quelle que soit la sortie (Ctrl+C, SIGTERM/fermeture du terminal, crash du conteneur, erreur), le conteneur
est arrete proprement (`docker stop`) puis supprime (`docker rm -f`) : il n'en reste jamais en arriere-plan.
Le port est publie sur 127.0.0.1 uniquement (pas accessible depuis le reseau).
"""
from __future__ import annotations

import json
import re
import signal
import uuid
from typing import Callable, Optional

from . import config
from .tools.checks import healthcheck
from .tools.docker import DockerCLI, DockerError

EXIT_OK, EXIT_FAILED, EXIT_UNAVAILABLE = 0, 1, 2


def last_image(project_id: str) -> Optional[str]:
    """Reference de la derniere image VALIDEE (build + tests dans le conteneur) de ce projet, d'apres result.json."""
    try:
        data = json.loads((config.WORKSPACES_DIR / project_id / "result.json").read_text(encoding="utf-8"))
        dk = data.get("docker") or {}
        return dk.get("image") if dk.get("ok") else None
    except (OSError, ValueError, AttributeError):
        return None


class _Stop(KeyboardInterrupt):
    """SIGTERM / SIGHUP (fermeture du terminal, `kill`) traites comme un Ctrl+C."""


def _install_signal_handlers() -> dict:
    def handler(signum, frame):
        raise _Stop()
    saved = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)          # SIGHUP n'existe pas sous Windows
        if sig is not None:
            try:
                saved[sig] = signal.signal(sig, handler)
            except ValueError:                     # pas dans le thread principal : on laisse tel quel
                pass
    return saved


def serve_container(image: str, host_port: int = 8000, *, docker: Optional[DockerCLI] = None,
                    on_line: Callable[[str], None] = print, say: Callable[[str], None] = print,
                    health_timeout_s: int = 60) -> int:
    """Demarre le conteneur, affiche l'URL puis ses logs en direct jusqu'a Ctrl+C. Retourne un code de sortie."""
    d = docker or DockerCLI()
    ok, info = d.available()
    if not ok:
        say("[ECHEC ] " + info)
        return EXIT_UNAVAILABLE
    try:
        d.image_id(image)
    except DockerError:
        say(f"[ECHEC ] image introuvable : {image}. Construire d'abord l'image : python -m agent_api_docker <projet> --docker")
        return EXIT_UNAVAILABLE
    port = d.exposed_port(image) or 8000
    name = f"agent-serve-{re.sub(r'[^a-z0-9_.-]+', '-', image.split(':')[0].lower())[:30]}-{uuid.uuid4().hex[:6]}"

    saved = _install_signal_handlers()
    cid: Optional[str] = None
    code = EXIT_OK
    try:
        say(f"Demarrage du conteneur depuis {image} (port {host_port} -> {port})...")
        run = d.run(image, port, name=name, host_port=host_port)
        cid = run.get("container_id")
        if run["exit_code"] != 0 or not run.get("host_port"):
            logs = (run["logs"] + ("\n" + d.logs(cid) if cid else "")).strip()
            say("[ECHEC ] le conteneur n'a pas demarre :\n" + logs[-800:])
            if re.search(r"port is already allocated|address already in use", logs, re.I):
                say(f"Le port {host_port} est deja utilise : relancer avec --port <autre port> (ex. --port {host_port + 1}).")
            return EXIT_FAILED
        base = f"http://127.0.0.1:{run['host_port']}"
        hc = healthcheck(base + "/health", health_timeout_s, alive=lambda: d.state(cid)["status"] == "running")
        if not hc["ok"]:
            say(f"[ECHEC ] /health ne repond pas : {hc['detail']}\n--- logs du conteneur ---\n" + d.logs(cid, 40))
            return EXIT_FAILED
        say(f"\nAPI en ligne : {base}/docs   (Ctrl+C = arreter et supprimer le conteneur)\n")
        d.follow_logs(cid, on_line)                 # bloque jusqu'a Ctrl+C ou arret du conteneur
        st = d.state(cid)
        if st["status"] != "running":
            say(f"[ECHEC ] le conteneur s'est arrete de lui-meme (code {st['exit_code']}).\n" + d.logs(cid, 40))
            code = EXIT_FAILED
    except KeyboardInterrupt:
        say("\nArret demande : arret du conteneur...")
    finally:
        if cid:
            try:
                d.stop(cid)                         # arret propre ; un 2e Ctrl+C passe directement a la suppression forcee
            except KeyboardInterrupt:
                pass
            d.cleanup(cid)
            say(f"Conteneur {name} arrete et supprime.")
        for sig, previous in saved.items():
            signal.signal(sig, previous)
    return code
