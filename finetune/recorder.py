"""Enregistre CHAQUE appel LLM d'un agent et reconstruit la trajectoire complète (ce que le modèle a vu et répondu).

La boucle d'agent « compacte » les anciennes sorties d'outils pour tenir dans la limite de tokens. Pour entraîner, on veut ce que le modèle a
VU au moment de décider : on reprend donc chaque message à sa PREMIÈRE apparition (avant compaction). Une trajectoire dont des sorties sont
quand même archivées est signalée (`compacted`) et exclue par défaut de l'export.
"""
from __future__ import annotations

import copy
import os
import re

KEEP = ("role", "content", "tool_calls", "tool_call_id", "name")


class RecordingClient:
    """Enveloppe un client LLM (Groq, secours, cassettes, script) sans changer son comportement."""

    def __init__(self, inner):
        self.inner = inner
        self.records: list[dict] = []
        self.tools = None

    def __getattr__(self, name):                       # model, tpm_limit, prompt_estimate... du client enveloppé
        return getattr(self.inner, name)

    def chat(self, messages, tools=None):
        resp = self.inner.chat(messages, tools)
        if self.tools is None and tools:
            self.tools = copy.deepcopy(tools)
        self.records.append({"messages": copy.deepcopy(messages), "response": copy.deepcopy(resp.message),
                             "usage": dict(resp.usage or {}), "model": resp.model})
        return resp


def sanitize(msg: dict) -> dict:
    """Format de chat standard : rôle, contenu, appels d'outils. Retire les champs propres au fournisseur (raisonnement, etc.)."""
    out = {k: msg[k] for k in KEEP if k in msg and msg[k] is not None}
    out.setdefault("role", "assistant")                # un message du modèle sans rôle (faux LLM de test) est un message assistant
    out.setdefault("content", "")
    if out.get("content") is None:
        out["content"] = ""
    if out.get("tool_calls"):
        out["tool_calls"] = [{"id": c.get("id"), "type": "function",
                              "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments", "{}")}} for c in out["tool_calls"]]
    return out


def rebuild_trajectory(records: list[dict]) -> dict | None:
    """-> {"messages": [...], "tools": [...], "steps": n, "compacted": bool, "models": [...]} ou None si aucun appel n'a abouti."""
    if not records:
        return None
    last = records[-1]
    n = len(last["messages"])
    msgs = []
    for p in range(n):
        first = next(r for r in records if len(r["messages"]) > p)           # 1re apparition du message p : version non compactée
        msgs.append(first["messages"][p])
    msgs.append(last["response"])
    flat = [sanitize(m) for m in msgs]
    compacted = any((m["role"] == "tool" and m["content"].startswith("[archivé")) or
                    any("archivé" in (c["function"]["arguments"] or "") for c in m.get("tool_calls") or []) for m in flat)
    return {"messages": flat, "tools": None, "steps": len(records), "compacted": compacted,
            "models": sorted({r["model"] for r in records if r.get("model")}),
            "tokens": sum(int((r["usage"] or {}).get("total_tokens", 0)) for r in records)}


SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)


def secret_values(environ=None) -> list[str]:
    """Valeurs de variables d'environnement qui ressemblent à des secrets (clés d'API) : à ne JAMAIS laisser dans un jeu de données."""
    env = os.environ if environ is None else environ
    return [v for k, v in env.items() if SECRET_NAME.search(k) and len(v) >= 12]


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        text = text.replace(s, "[REDACTED]")
    return text
