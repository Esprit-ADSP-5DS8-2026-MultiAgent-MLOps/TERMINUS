"""Publication Docker Hub. REGLE : la verification d'approbation est faite ICI, en code, jamais par le LLM.

L'approbation est liee a l'IDENTIFIANT de l'image (sha256:...), pas a un nom ou un tag : si l'image locale change
entre l'approbation et le push (nouveau build, tag deplace), le push est refuse.
Le token Docker Hub n'est jamais passe en argument ni ecrit dans les logs (voir DockerCLI.login).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .. import config
from .docker import DockerCLI


class ApprovalRequired(Exception):
    pass


class PushError(Exception):
    """Login ou push refuse par Docker Hub (identifiants, droits, reseau) : probleme d'environnement, pas de fichier genere."""


class LoginError(PushError):
    """Identifiants refuses par Docker Hub : l'utilisateur peut en ressaisir (le push n'a pas commence)."""


_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

_PUSH_HINTS = [
    (r"denied|unauthorized|authentication required|requested access",
     "acces refuse : verifier que le token a les droits Read & Write et que l'espace de nom correspond a votre "
     "compte ou a une organisation dont vous etes membre (DOCKER_IMAGE_NAMESPACE)"),
    (r"toomanyrequests|rate limit", "limite de requetes Docker Hub atteinte : reessayer plus tard"),
    (r"no space left", "disque plein cote Docker"),
    (r"timeout|TLS|no such host|connection reset|EOF", "probleme reseau : verifier la connexion puis relancer"),
]


def _hint(logs: str) -> str:
    return next((h for pat, h in _PUSH_HINTS if re.search(pat, logs, re.I)), "")


def docker_push(image: str, tag: str | list[str], approved_digest: str | None, actual_digest: str, *,
                source_ref: str, docker: Optional[DockerCLI] = None, username: Optional[str] = None,
                token: Optional[str] = None, on_line: Callable[[str], None] = lambda s: None) -> dict:
    """Pousse source_ref vers image:tag (un ou plusieurs tags). Retourne {repository, tags, digest}.

    approved_digest : identifiant approuve (par l'utilisateur, via le code) ; actual_digest : identifiant de l'image testee.
    """
    if not approved_digest or approved_digest != actual_digest:
        raise ApprovalRequired("Push refuse : pas d'approbation valide pour ce digest d'image.")
    tags = [tag] if isinstance(tag, str) else list(tag)
    if not _IMAGE_RE.match(image) or not tags or not all(_TAG_RE.match(t) for t in tags):
        raise ValueError(f"Nom d'image ou tag invalide : {image}:{','.join(tags)} (attendu : espace_de_nom/depot:tag)")

    d = docker or DockerCLI()
    current = d.image_id(source_ref)                 # relu ICI : l'image locale ne doit pas avoir change
    if current != approved_digest:
        raise ApprovalRequired(f"Push refuse : l'image locale a change depuis l'approbation ({current[:19]} != {approved_digest[:19]}).")

    user, secret = username or config.DOCKERHUB_USERNAME, token or config.DOCKERHUB_TOKEN
    if not user or not secret:
        raise PushError("Identifiants Docker Hub absents : renseigner DOCKERHUB_USERNAME et DOCKERHUB_TOKEN dans .env "
                        "(token a creer sur hub.docker.com > Account settings > Personal access tokens, droits Read & Write).")
    login = d.login(user, secret)
    if login["exit_code"] != 0:
        raise LoginError("Connexion a Docker Hub refusee : " + (_hint(login["logs"]) or login["logs"][-300:]))
    digest = None
    try:
        for t in tags:
            ref = f"{image}:{t}"
            d.tag(source_ref, ref)
            res = d.push(ref, on_line=on_line)
            if res["exit_code"] != 0:
                raise PushError(f"Push de {ref} echoue : " + (_hint(res["logs"]) or res["logs"][-300:]))
            digest = res["digest"] or digest
    finally:
        d.logout()
    return {"repository": image, "tags": tags, "digest": digest}
