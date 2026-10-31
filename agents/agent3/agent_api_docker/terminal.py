"""Saisie des identifiants Docker Hub dans le terminal (par le CODE, jamais par le LLM).

Le nom d'utilisateur est demande en clair (valeur par defaut = DOCKERHUB_USERNAME de .env, s'il existe) ; le token est
lu SANS ECHO (getpass). Rien n'est ecrit sur disque ni dans les traces : la paire reste en memoire le temps du push.
"""
from __future__ import annotations

import getpass
from typing import Callable, Optional


def make_credentials_prompt(default_user: str = "", input_fn: Callable[[str], str] = input,
                            secret_fn: Callable[[str], str] = getpass.getpass,
                            out: Callable[[str], None] = print) -> Callable[[Optional[str]], Optional[tuple[str, str]]]:
    """Retourne une fonction `ask(raison) -> (utilisateur, token) | None` (None = l'utilisateur annule).

    1er appel (raison=None) : demande utilisateur + token. Appels suivants (connexion refusee) : redemande le token
    seulement, pour le meme compte.
    """
    state: dict = {"user": None}

    def ask(reason: Optional[str] = None) -> Optional[tuple[str, str]]:
        try:
            if reason is None:
                out("\nConnexion a Docker Hub avant le push (le token n'est ni affiche ni enregistre).")
                label = f"Nom d'utilisateur Docker Hub [{default_user}] : " if default_user else "Nom d'utilisateur Docker Hub : "
                user = input_fn(label).strip() or default_user
                if not user:
                    out("Aucun nom d'utilisateur : push annule.")
                    return None
                state["user"] = user
            else:
                out(reason)
                out(f"Nouvelle tentative pour le compte '{state['user']}' (token vide = annuler).")
            token = secret_fn("Token Docker Hub (Personal Access Token, droits Read & Write) : ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            return None
        if not token:
            out("Aucun token : push annule.")
            return None
        return state["user"], token

    return ask
