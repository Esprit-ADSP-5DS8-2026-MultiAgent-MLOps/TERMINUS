"""
Client LLM : un seul protocole (chat/completions d'OpenAI) pour Groq, Gemini et Ollama.

Deux rôles, deux familles de modèles :
- DÉCIDEUR (Groq) : enquête avec des outils, diagnostique, propose une action ;
- JUGE (Gemini)   : relit et note. Un modèle d'une autre famille ne partage pas les mêmes biais.

Configuration : Monitoring_Agent/.env (jamais commité), voir .env.example.
Le .env est lu dans un dict, pas injecté dans os.environ (les clés ne fuient pas vers les sous-processus).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from ..datasets import PACKAGE_ROOT

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
DEFAULTS = {
    "MONITORING_DECIDER": "groq:llama-3.3-70b-versatile",
    "MONITORING_DECIDER_FALLBACK": "gemini:gemini-2.5-flash",
    "MONITORING_JUDGE": "gemini:gemini-2.5-flash",
}

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """429, 5xx, réseau : on peut basculer sur un autre modèle."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    raw_message: dict = field(default_factory=dict)


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
             temperature: float = 0.0) -> LLMResponse: ...


def _settings() -> dict[str, str]:
    env_file = Path(os.environ.get("MONITORING_ENV_FILE", PACKAGE_ROOT / ".env"))
    values = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.exists() else {}
    return {**values, **{k: v for k, v in os.environ.items() if k.startswith(("MONITORING_", "GROQ_", "GEMINI_"))}}


class OpenAICompatModel:
    def __init__(self, provider: str, model: str, api_key: str | None, base_url: str | None = None,
                 timeout_s: float = 60, max_tokens: int = 2048, retries: int = 2, max_wait_s: float = 20):
        self.provider, self.model = provider, model
        self.name = f"{provider}:{model}"
        self.base_url = (base_url or PROVIDERS[provider][0]).rstrip("/")
        self.api_key, self.timeout_s, self.max_tokens = api_key, timeout_s, max_tokens
        self.retries, self.max_wait_s = retries, max_wait_s

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                "max_tokens": self.max_tokens}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s)
            except httpx.HTTPError as exc:
                last, wait = f"réseau : {exc}", 2.0 * (attempt + 1)
            else:
                if r.status_code == 200:
                    return self._parse(r.json())
                last = f"HTTP {r.status_code} : {r.text[:300]}"
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    raise LLMError(f"{self.name} — {last}")          # clé invalide, modèle inconnu... : définitif
                wait = _retry_after(r) or 2.0 * (attempt + 1)
            if attempt == self.retries or wait > self.max_wait_s:
                break
            time.sleep(wait)
        raise LLMUnavailable(f"{self.name} — {last}")

    def _parse(self, data: dict) -> LLMResponse:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"réponse inattendue de {self.name} : {str(data)[:200]}") from exc
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_invalid_json": raw}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", tc["function"]["name"], args))
        clean = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            clean["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in calls]
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls, model=self.name, raw_message=clean)


def _retry_after(r: httpx.Response) -> float:
    h = r.headers.get("retry-after")
    if h:
        try:
            return float(h)
        except ValueError:
            pass
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)(ms|s)", r.text or "", re.IGNORECASE)
    if m:
        return float(m.group(2)) / (1000 if m.group(3).lower() == "ms" else 1) + 60 * int(m.group(1) or 0)
    return 0.0


class FallbackModel:
    """Essaie les modèles dans l'ordre ; bascule sur le suivant si l'un est indisponible."""

    def __init__(self, models: list[ChatModel]):
        self.models = models
        self.name = " -> ".join(m.name for m in models)
        self.last_used = ""

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        errors = []
        for m in self.models:
            try:
                resp = m.chat(messages, tools, json_mode, temperature)
                self.last_used = m.name
                return resp
            except LLMUnavailable as exc:
                errors.append(str(exc))
        raise LLMUnavailable(" || ".join(errors))


def build_model(spec: str, settings: dict[str, str]) -> OpenAICompatModel | None:
    """« groq:llama-3.3-70b-versatile » -> client, ou None si la clé manque."""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise LLMError(f"modèle mal défini : {spec!r} (attendu fournisseur:modèle, fournisseurs {list(PROVIDERS)})")
    key_env = PROVIDERS[provider][1]
    key = settings.get(key_env) if key_env else None
    if key_env and not key:
        return None
    return OpenAICompatModel(provider, model, key, settings.get(f"MONITORING_{provider.upper()}_BASE_URL"))


def decider_and_judge() -> tuple[ChatModel | None, ChatModel | None]:
    s = _settings()
    primary = build_model(s.get("MONITORING_DECIDER", DEFAULTS["MONITORING_DECIDER"]), s)
    fallback = build_model(s.get("MONITORING_DECIDER_FALLBACK", DEFAULTS["MONITORING_DECIDER_FALLBACK"]), s)
    chain = [m for m in (primary, fallback) if m is not None]
    decider = FallbackModel(chain) if len(chain) > 1 else (chain[0] if chain else None)
    judge = build_model(s.get("MONITORING_JUDGE", DEFAULTS["MONITORING_JUDGE"]), s)
    return decider, judge


# ---------------------------------------------------------------------------
# Sortie structurée : JSON -> Pydantic, avec une seconde chance si c'est invalide
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("aucun objet JSON dans la réponse")
    return json.loads(text[start:end + 1])


def structured(model: ChatModel, messages: list[dict], schema: type[T], temperature: float = 0.0,
               attempts: int = 2) -> T:
    msgs = list(messages)
    last_error = ""
    for _ in range(attempts):
        resp = model.chat(msgs, json_mode=True, temperature=temperature)
        try:
            return schema.model_validate(extract_json(resp.text))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:600]
            msgs = msgs + [{"role": "assistant", "content": resp.text},
                           {"role": "user", "content": f"Ta réponse est invalide : {last_error}\n"
                                                       "Renvoie UNIQUEMENT l'objet JSON corrigé, conforme au schéma."}]
    raise LLMError(f"sortie invalide après {attempts} essais : {last_error}")


def list_models(provider: str) -> list[str]:
    """Modèles disponibles chez un fournisseur (commande `llm-check`)."""
    s = _settings()
    base, key_env = PROVIDERS[provider]
    key = s.get(key_env) if key_env else None
    if key_env and not key:
        raise LLMError(f"{key_env} absente de Monitoring_Agent/.env")
    r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=20)
    r.raise_for_status()
    return sorted(m["id"].removeprefix("models/") for m in r.json().get("data", []))
