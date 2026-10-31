"""Client LLM minimal avec appel d'outils (function calling).

Compatible avec tout serveur qui parle le protocole `chat/completions` d'OpenAI :
OpenAI, Groq, Gemini (endpoint compatible OpenAI), Ollama, vLLM...
Lit sa configuration dans config.py (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY_DOCKER, LLM_BASE_URL).

Le reste du code ne depend que du protocole `LLMClient` : en test on injecte un faux LLM.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from . import config

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}


class LLMError(Exception):
    """Appel au LLM impossible (reseau, quota, reponse invalide)."""


class LLMUnavailable(LLMError):
    """Fournisseur momentanement indisponible (429 / 5xx / reseau) : on peut basculer sur un autre."""

    def __init__(self, msg: str, retry_after_s: float = 0.0):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class LLMNotConfigured(LLMError):
    """LLM_PROVIDER / LLM_MODEL / cle absents."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra: dict | None = None     # `extra_content` du fournisseur (Gemini 3 : thought_signature), a renvoyer tel quel


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""      # modele qui a reellement repondu (utile en cas de bascule)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def _retry_after_s(r: httpx.Response) -> float:
    """Delai demande par le fournisseur : en-tete Retry-After, sinon 'try again in 1.5s' / '350ms'."""
    h = r.headers.get("retry-after")
    if h:
        try:
            return float(h)
        except ValueError:
            pass
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)(ms|s)", r.text or "", re.I)
    if m:
        secs = float(m.group(2)) / (1000 if m.group(3).lower() == "ms" else 1)
        return secs + 60 * int(m.group(1) or 0)
    return 0.0


# Gemini 3 (endpoint compatible OpenAI) EXIGE un `thought_signature` sur chaque appel d'outil de l'historique, sinon HTTP 400
# "Function call is missing a thought_signature". Les appels venant d'un autre modele (bascule qwen -> gemini en cours de
# boucle) n'en ont pas : Google documente la valeur ci-dessous pour ce cas. Les autres fournisseurs rejettent a l'inverse
# les champs inconnus : on retire `extra_content` pour eux.
GEMINI_DUMMY_SIGNATURE = "skip_thought_signature_validator"


def _is_gemini(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in base_url


def prepare_messages(messages: list[dict], gemini: bool) -> list[dict]:
    """Copie de l'historique adaptee au fournisseur (l'historique d'origine n'est jamais modifie)."""
    out = []
    for m in messages:
        calls = m.get("tool_calls")
        if m.get("role") != "assistant" or not calls:
            out.append(m)
            continue
        fixed = []
        for tc in calls:
            tc = dict(tc)
            if gemini:
                extra = dict(tc.get("extra_content") or {})
                google = dict(extra.get("google") or {})
                google.setdefault("thought_signature", GEMINI_DUMMY_SIGNATURE)
                extra["google"] = google
                tc["extra_content"] = extra
            else:
                tc.pop("extra_content", None)
            fixed.append(tc)
        out.append({**m, "tool_calls": fixed})
    return out


class OpenAICompatClient:
    def __init__(self, model: str, base_url: str, api_key: str = "", timeout_s: int = 60,
                 max_tokens: int = 4096, temperature: float = 0.0, retries: int = 2,
                 max_wait_s: float = 15.0):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key, self.timeout_s = api_key, timeout_s
        self.max_tokens, self.temperature, self.retries = max_tokens, temperature, retries
        self.max_wait_s = max_wait_s   # au-dela, on n'attend pas : on rend la main (bascule possible)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {"model": self.model, "messages": prepare_messages(messages, _is_gemini(self.base_url)),
                                "temperature": self.temperature,
                                "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        last_err, wait_hint = "", 0.0
        for attempt in range(self.retries + 1):
            wait = 0.0
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout_s)
            except httpx.HTTPError as e:
                last_err = f"erreur reseau : {e}"
                wait = 1.5 * (2 ** attempt)
            else:
                if r.status_code == 200:
                    return self._parse(r.json())
                last_err = f"HTTP {r.status_code} : {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(f"Appel LLM echoue ({self.base_url}) - {last_err}")  # definitif : cle, modele...
                wait_hint = _retry_after_s(r)
                wait = wait_hint or 1.5 * (2 ** attempt)
            if attempt == self.retries:
                break
            if wait > self.max_wait_s:   # quota minute epuise : inutile de bloquer, on laisse la bascule agir
                break
            time.sleep(wait + random.uniform(0, 0.5))
        raise LLMUnavailable(f"Appel LLM echoue ({self.base_url}) - {last_err}", wait_hint)

    @staticmethod
    def _parse(data: dict) -> LLMResponse:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Reponse LLM inattendue : {str(data)[:300]}") from e
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_invalid_json": raw}  # l'agent le signalera au LLM, qui corrigera
            calls.append(ToolCall(id=tc.get("id") or f"call_{len(calls)}", name=tc["function"]["name"], arguments=args,
                                  extra=tc.get("extra_content")))
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls, model=data.get("model") or "")


class FallbackClient:
    """Essaie le LLM principal ; s'il est indisponible (429/5xx/reseau), bascule sur le suivant.

    Un fournisseur en panne est mis en 'pause' (coupe-circuit) le temps indique par son Retry-After
    (60 s par defaut) : on ne perd plus de temps a le re-solliciter a chaque etape, puis on le reessaie.
    L'historique des messages est identique pour tous les fournisseurs (protocole OpenAI) :
    la bascule peut donc avoir lieu au milieu d'une boucle d'agent.
    """

    def __init__(self, clients: list[OpenAICompatClient], cooldown_s: float = 60.0,
                 on_switch=lambda msg: None):
        self.clients, self.cooldown_s, self.on_switch = clients, cooldown_s, on_switch
        self._paused_until = [0.0] * len(clients)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        now = time.monotonic()
        order = [i for i in range(len(self.clients)) if self._paused_until[i] <= now]
        order += [i for i in range(len(self.clients)) if i not in order]   # tous en pause : on retente quand meme
        errors: list[str] = []
        for rank, i in enumerate(order):
            c = self.clients[i]
            try:
                resp = c.chat(messages, tools)
            except LLMUnavailable as e:
                self._paused_until[i] = time.monotonic() + (e.retry_after_s or self.cooldown_s)
                errors.append(str(e))
                if rank + 1 < len(order):
                    self.on_switch(f"{c.model} indisponible -> bascule sur {self.clients[order[rank + 1]].model}")
                continue
            resp.model = resp.model or c.model
            return resp
        raise LLMError("Tous les LLM sont indisponibles : " + " || ".join(errors))


def _build(model: str, base_url: str, key: str) -> OpenAICompatClient:
    return OpenAICompatClient(model, base_url, key, config.LLM_TIMEOUT_S, config.LLM_MAX_TOKENS,
                              config.LLM_TEMPERATURE, max_wait_s=config.LLM_MAX_WAIT_S)


def make_client() -> LLMClient:
    """Construit le client depuis la configuration (+ LLM de secours optionnel), ou explique ce qui manque."""
    provider = config.LLM_PROVIDER.lower()
    base_url = config.LLM_BASE_URL or DEFAULT_BASE_URLS.get(provider, "")
    missing = []
    if not config.LLM_MODEL:
        missing.append("LLM_MODEL")
    if not base_url:
        missing.append("LLM_PROVIDER (openai|groq|gemini|ollama) ou LLM_BASE_URL")
    if provider != "ollama" and not config.LLM_BASE_URL and not config.LLM_API_KEY:
        missing.append(config.LLM_API_KEY_ENV)
    if missing:
        raise LLMNotConfigured("LLM non configure, a renseigner dans .env : " + ", ".join(missing)
                               + "  (verifier avec : python -m agent_api_docker.config)")
    primary = _build(config.LLM_MODEL, base_url, config.LLM_API_KEY)

    clients = [primary]
    for fb in config.LLM_FALLBACKS:
        fb_provider = fb["provider"].lower()
        fb_url = fb["base_url"] or DEFAULT_BASE_URLS.get(fb_provider, "")
        if not fb_url:
            raise LLMNotConfigured(f"LLM de secours n°{fb['n']} : renseigner LLM_FALLBACK_{fb['n']}_PROVIDER ou _BASE_URL")
        if fb_provider != "ollama" and not fb["base_url"] and not fb["api_key"]:
            raise LLMNotConfigured(f"LLM de secours n°{fb['n']} incomplet : renseigner {fb['key_env']} dans .env")
        clients.append(_build(fb["model"], fb_url, fb["api_key"]))
    if len(clients) == 1:
        return primary                      # pas de secours configure : comportement inchange
    return FallbackClient(clients, on_switch=lambda m: print(f"  [llm] {m}", flush=True))