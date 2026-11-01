"""Clients LLM : Groq (API compatible OpenAI), client scripté (tests) et cache enregistrement/rejeu.

Tous exposent `chat(messages, tools) -> LLMResponse`. La clé API n'est jamais écrite dans un message d'erreur.
Les erreurs sont typées (`LLMError.kind`) pour que la boucle d'agent les transforme en STATUTS, pas en exceptions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(Exception):
    """kind : quota | timeout | network | http | auth | bad_response | missing_key | cassette_missing"""

    def __init__(self, message, kind: str = "http"):
        super().__init__(message)
        self.kind = kind


@dataclass
class LLMResponse:
    message: dict
    usage: dict = field(default_factory=dict)       # prompt_tokens, completion_tokens, total_tokens, cached_tokens
    latency_s: float = 0.0
    model: str = ""
    cached: bool = False                            # True : réponse rejouée depuis une cassette (pas un cache côté Groq)


def parse_duration(text) -> float | None:
    """'7.66s' | '1m30.5s' | '250ms' -> secondes (format des en-têtes x-ratelimit-reset-*)."""
    if not text:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|m|s)", str(text).strip())
    if not parts:
        return None
    return sum(float(v) * {"ms": 0.001, "s": 1.0, "m": 60.0}[u] for v, u in parts)


# Un modèle différent par agent. Le Judge utilise une autre famille que le Repair : réduit le biais d'un modèle
# qui approuve ses propres sorties. Vérifiez les noms sur console.groq.com/docs/models.
AGENT_DEFAULTS = {
    "validator": {"model": "openai/gpt-oss-120b", "reasoning_effort": "low", "temperature": 0.0, "max_completion_tokens": 2500},
    "repair": {"model": "openai/gpt-oss-120b", "reasoning_effort": "low", "temperature": 0.0, "max_completion_tokens": 2500},
    "judge": {"model": "llama-3.3-70b-versatile", "reasoning_effort": None, "temperature": 0.0, "max_completion_tokens": 2000},
}


def agent_settings(agent: str, model: str | None = None) -> dict:
    """Priorité : argument > GROQ_MODEL_<AGENT> > GROQ_MODEL (validator/repair seulement) > défaut de l'agent."""
    s = dict(AGENT_DEFAULTS[agent])
    env = os.environ.get(f"GROQ_MODEL_{agent.upper()}") or (os.environ.get("GROQ_MODEL") if agent != "judge" else None)
    s["model"] = model or env or s["model"]
    effort = os.environ.get(f"GROQ_REASONING_EFFORT_{agent.upper()}")
    if effort:
        s["reasoning_effort"] = None if effort.lower() == "none" else effort
    return s


class GroqClient:
    """Client Groq (/chat/completions) avec appel d'outils.

    - température configurable (0 par défaut : réduit la variabilité, ne l'élimine pas) ;
    - `reasoning_effort` envoyé seulement aux modèles qui le supportent (famille openai/gpt-oss) ;
    - nouvelles tentatives avec attente progressive sur 429/5xx (respecte `retry-after`) ;
    - cadence : le quota de tokens/minute restant est comparé à l'estimation de la requête, EN DÉDUISANT les
      tokens déjà en cache (le préfixe identique renvoyé à chaque étape) : l'ancienne estimation était trop prudente.
    `post` et `sleep` sont injectables pour tester sans réseau.
    """
    BASE_URL = "https://api.groq.com/openai/v1"
    CHARS_PER_TOKEN = 3.5          # estimation pour la cadence (attente)
    GUARD_CHARS_PER_TOKEN = 3.0    # estimation PRUDENTE pour le plafond dur (on préfère surestimer)
    MIN_COMPLETION = 700           # en dessous, une réponse (rapport complet) ne tient plus
    SAFETY_MARGIN = 150
    DEFAULT_TPM_LIMIT = 8000       # palier gratuit « on_demand » ; GROQ_TPM_LIMIT=0 désactive ; à régler selon votre palier/modèle

    def __init__(self, api_key=None, model=None, *, agent: str | None = None, temperature=None, reasoning_effort="__default__",
                 max_completion_tokens=None, timeout=90, max_retries=5, base_url=None, post=None, sleep=time.sleep, tpm_limit="__default__",
                 max_retry_wait=65.0, max_tokens_key="max_completion_tokens"):
        d = agent_settings(agent or "validator", model)
        self.agent = agent
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = d["model"]
        self.temperature = d["temperature"] if temperature is None else temperature
        self.reasoning_effort = d["reasoning_effort"] if reasoning_effort == "__default__" else reasoning_effort
        self.max_completion_tokens = max_completion_tokens or d["max_completion_tokens"]
        self.timeout, self.max_retries = timeout, max_retries
        self.max_retry_wait = max_retry_wait            # attente max sur 429/5xx ; au-delà on rend la main (le secours prend le relais)
        self.max_tokens_key = max_tokens_key            # "max_completion_tokens" (Groq/OpenAI) ou "max_tokens" (Gemini, Ollama)
        if tpm_limit == "__default__":
            env = os.environ.get("GROQ_TPM_LIMIT")
            tpm_limit = int(env) if env and env.isdigit() else self.DEFAULT_TPM_LIMIT
        self.tpm_limit = tpm_limit or None             # tokens/minute autorisés pour CE modèle (requête unique incluse)
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self._post, self._sleep = post, sleep
        self._remaining_tokens = None
        self._reset_s = None
        self._last_prompt_tokens = 0
        self._last_cached_tokens = 0
        self._ratio = None                              # caractères par token, mesuré

    @classmethod
    def for_agent(cls, agent: str, model: str | None = None, **kw):
        return cls(agent=agent, model=model, **kw)

    # ------------------------------------------------------------------ requête
    def build_payload(self, messages, tools=None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   self.max_tokens_key: self.max_completion_tokens}
        if self.reasoning_effort and self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        return payload

    def _do_post(self, url, headers, payload):
        if self._post:
            return self._post(url, headers, payload, self.timeout)
        import requests
        return requests.post(url, headers=headers, json=payload, timeout=self.timeout)

    def prompt_estimate(self, messages, tools=None) -> int:
        """Estimation PRUDENTE (par excès) des tokens d'entrée d'une requête."""
        chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))
        # avant le 1er appel : ratio prudent ; ensuite : ratio MESURÉ sur les `usage` réels de Groq, avec 8 % de marge
        ratio = self._ratio * 0.92 if self._ratio else self.GUARD_CHARS_PER_TOKEN
        return int(chars / ratio)

    def _guard_request_size(self, payload):
        """Plafond dur : prompt + réponse maximale doit tenir dans la limite TPM, sinon Groq répond 413 (attendre n'y change rien).
        On réduit donc `max_completion_tokens` ; si même le minimum ne tient pas, on lève `too_large` AVANT l'appel (la boucle compacte le contexte)."""
        if not self.tpm_limit:
            return
        prompt = self.prompt_estimate(payload["messages"], payload.get("tools"))
        room = self.tpm_limit - prompt - self.SAFETY_MARGIN
        if room < self.MIN_COMPLETION:
            raise LLMError(f"requête estimée à ~{prompt} tokens d'entrée pour une limite de {self.tpm_limit} tokens/minute : "
                           "contexte à réduire", "too_large")
        payload[self.max_tokens_key] = int(min(self.max_completion_tokens, room))

    def estimate_quota_tokens(self, payload) -> float:
        """Tokens qui compteront contre le quota : prompt estimé - préfixe déjà en cache + sortie maximale."""
        prompt_est = len(json.dumps(payload["messages"], ensure_ascii=False) + json.dumps(payload.get("tools") or [])) / self.CHARS_PER_TOKEN
        # la conversation ne fait que grandir : le préfixe de l'appel précédent est (normalement) en cache
        cached_est = min(self._last_cached_tokens, prompt_est) if prompt_est >= self._last_prompt_tokens else 0
        return max(prompt_est - cached_est, 0) + self.max_completion_tokens

    def _pace(self, estimated_tokens: float):
        if self._remaining_tokens is not None and estimated_tokens > self._remaining_tokens and self._reset_s:
            self._sleep(min(self._reset_s + 0.5, 65))

    def _read_limits(self, headers):
        try:
            rem = headers.get("x-ratelimit-remaining-tokens")
            self._remaining_tokens = float(rem) if rem is not None else None
            self._reset_s = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        except Exception:
            self._remaining_tokens = self._reset_s = None

    @staticmethod
    def _retry_after(headers, attempt: int) -> float:
        raw = headers.get("retry-after")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return parse_duration(raw) or min(2 ** attempt, 30)

    def chat(self, messages, tools=None) -> LLMResponse:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY manquante (variable d'environnement)", "missing_key")
        payload = self.build_payload(messages, tools)
        self._guard_request_size(payload)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self._pace(self.estimate_quota_tokens(payload))
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._do_post(url, headers, payload)
            except Exception as e:  # réseau : on ne met JAMAIS la clé dans le message
                kind = "timeout" if "timeout" in type(e).__name__.lower() else "network"
                if attempt < self.max_retries:
                    self._sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"erreur réseau ({type(e).__name__})", kind) from None
            self._read_limits(resp.headers)
            code = resp.status_code
            if code == 200:
                try:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                except Exception:
                    raise LLMError("réponse Groq illisible", "bad_response") from None
                u = data.get("usage") or {}
                cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
                usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                         "total_tokens": u.get("total_tokens", u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)),
                         "cached_tokens": cached}
                self._last_prompt_tokens, self._last_cached_tokens = usage["prompt_tokens"], cached
                if usage["prompt_tokens"] > 200:
                    sent = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))
                    self._ratio = max(2.0, min(sent / usage["prompt_tokens"], 6.0))
                return LLMResponse(message=msg, usage=usage, latency_s=time.time() - t0, model=data.get("model", self.model))
            if code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                wait = min(self._retry_after(resp.headers, attempt), 65)
                if wait <= self.max_retry_wait:         # attente raisonnable : on réessaie ; sinon on échoue vite (bascule possible)
                    self._sleep(wait)
                    continue
            body = (getattr(resp, "text", "") or "")[:300]
            if code == 413:
                m = re.search(r"Limit (\d+), Requested (\d+)", body)
                if m:
                    self.tpm_limit = int(m.group(1))       # la vraie limite du palier : les prochains appels s'y adaptent
                raise LLMError(f"HTTP 413 : requête trop grande pour la limite TPM. {body}", "too_large")
            kind = "quota" if code == 429 else "auth" if code in (401, 403) else "timeout" if code in (408, 504) else "http"
            raise LLMError(f"HTTP {code} : {body}", kind)
        raise LLMError("nombre maximal de tentatives atteint", "http")


class ScriptedClient:
    """Faux LLM pour les tests : renvoie les messages d'un script, dans l'ordre (ou lève l'exception fournie)."""
    model = "scripted"

    def __init__(self, script, tokens_per_call: int = 200, cached_per_call: int = 0, model: str | None = None):
        self.script, self.tokens_per_call, self.cached_per_call, self.calls = list(script), tokens_per_call, cached_per_call, []
        if model:
            self.model = model

    def chat(self, messages, tools=None) -> LLMResponse:
        self.calls.append(len(messages))
        if not self.script:
            raise LLMError("script épuisé", "http")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(message=item, model=self.model,
                           usage={"prompt_tokens": self.tokens_per_call, "completion_tokens": 20,
                                  "total_tokens": self.tokens_per_call + 20, "cached_tokens": self.cached_per_call})


class CachedClient:
    """Enregistre les réponses du LLM sur disque et les rejoue (tests rapides, gratuits, reproductibles).
    mode : 'auto' (rejoue sinon appelle et enregistre) | 'replay' (n'appelle jamais le LLM)."""

    def __init__(self, inner, cache_dir, mode: str = "auto"):
        self.inner, self.dir, self.mode = inner, Path(cache_dir).expanduser(), mode
        self.model = getattr(inner, "model", "cached")
        self.hits = self.misses = 0

    @property
    def tpm_limit(self):
        return getattr(self.inner, "tpm_limit", None)

    def prompt_estimate(self, messages, tools=None):
        f = getattr(self.inner, "prompt_estimate", None)
        return f(messages, tools) if f else 0

    def _path(self, messages, tools):
        raw = json.dumps({"m": self.model, "messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
        return self.dir / (hashlib.sha256(raw.encode()).hexdigest() + ".json")

    def chat(self, messages, tools=None) -> LLMResponse:
        p = self._path(messages, tools)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            self.hits += 1
            return LLMResponse(message=d["message"], usage=d["usage"], latency_s=0.0, model=d["model"], cached=True)
        if self.mode == "replay":
            raise LLMError("cassette manquante (mode replay) : relancez sans --replay pour l'enregistrer", "cassette_missing")
        resp = self.inner.chat(messages, tools)
        self.misses += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"message": resp.message, "usage": resp.usage, "model": resp.model}, ensure_ascii=False), encoding="utf-8")
        return resp


PROVIDER_URLS = {"groq": "https://api.groq.com/openai/v1", "openai": "https://api.openai.com/v1",
                 "gemini": "https://generativelanguage.googleapis.com/v1beta/openai", "ollama": "http://localhost:11434/v1"}


class FallbackClient:
    """Essaie le LLM principal ; s'il échoue (quota, 413, panne, réseau, clé...), bascule sur le secours suivant.

    Coupe-circuit : un fournisseur en échec est mis en pause `cooldown_s` secondes, les étapes suivantes de la boucle
    vont directement au secours (pas de temps perdu à re-solliciter un quota épuisé), puis il est réessayé.
    L'historique est au format OpenAI pour tous : on peut donc basculer au milieu d'une boucle d'agent.
    """
    NO_SWITCH = {"cassette_missing"}

    def __init__(self, clients, cooldown_s: float = 60.0, on_switch=lambda msg: None, clock=time.monotonic):
        self.clients, self.cooldown_s, self.on_switch, self._clock = list(clients), cooldown_s, on_switch, clock
        self._paused_until = [0.0] * len(self.clients)
        self.last_model = ""

    def _active(self):
        now = self._clock()
        return next((c for c, t in zip(self.clients, self._paused_until) if t <= now), self.clients[0])

    @property
    def model(self):
        return self.last_model or self.clients[0].model

    @property
    def tpm_limit(self):                 # la limite du client qui va être utilisé (None pour un secours sans plafond)
        return getattr(self._active(), "tpm_limit", None)

    def prompt_estimate(self, messages, tools=None):
        f = getattr(self._active(), "prompt_estimate", None)
        return f(messages, tools) if f else 0

    def chat(self, messages, tools=None) -> LLMResponse:
        now = self._clock()
        order = [i for i in range(len(self.clients)) if self._paused_until[i] <= now]
        order += [i for i in range(len(self.clients)) if i not in order]       # tous en pause : on retente quand même
        errors, first = [], None
        for rank, i in enumerate(order):
            c = self.clients[i]
            try:
                resp = c.chat(messages, tools)
            except LLMError as e:
                if e.kind in self.NO_SWITCH:
                    raise
                first = first or e
                self._paused_until[i] = self._clock() + self.cooldown_s
                errors.append(f"{c.model}: {e}")
                if rank + 1 < len(order):
                    self.on_switch(f"{c.model} indisponible ({e.kind}) -> bascule sur {self.clients[order[rank + 1]].model}")
                continue
            self.last_model = resp.model or c.model
            return resp
        raise LLMError("tous les LLM ont échoué : " + " || ".join(errors), first.kind if first else "http")


def _fallback_specs(env=None) -> list[dict]:
    """FALLBACK_<n>_PROVIDER|MODEL|API_KEY|BASE_URL (n = 1..3). Une entrée existe dès que FALLBACK_<n>_MODEL est défini."""
    env = os.environ if env is None else env
    out = []
    for n in (1, 2, 3):
        g = lambda k: env.get(f"FALLBACK_{n}_{k}", "").strip()
        if g("MODEL"):
            out.append({"n": n, "provider": g("PROVIDER").lower(), "model": g("MODEL"), "api_key": g("API_KEY"), "base_url": g("BASE_URL")})
    return out


def build_fallback_client(agent: str, spec: dict, **kw) -> GroqClient:
    """Client OpenAI-compatible pour un secours : pas de plafond TPM Groq, pas de reasoning_effort, clé propre."""
    url = spec["base_url"] or PROVIDER_URLS.get(spec["provider"], "")
    if not url:
        raise LLMError(f"FALLBACK_{spec['n']}_PROVIDER (groq|openai|gemini|ollama) ou FALLBACK_{spec['n']}_BASE_URL manquant", "missing_key")
    local = spec["provider"] == "ollama"
    if not spec["api_key"] and not local:
        raise LLMError(f"FALLBACK_{spec['n']}_API_KEY manquante", "missing_key")
    if spec["provider"] == "groq":      # secours Groq (2e clé, autre modèle) : mêmes réglages que le principal (plafond TPM, reasoning_effort)
        return GroqClient(api_key=spec["api_key"], model=spec["model"], agent=agent, base_url=url, **kw)
    return GroqClient(api_key=spec["api_key"] or "local", model=spec["model"], agent=agent, base_url=url, reasoning_effort=None,
                      tpm_limit=0, max_tokens_key="max_completion_tokens" if spec["provider"] == "openai" else "max_tokens", **kw)


def build_client(agent: str, model=None, cache_dir=None, replay=False):
    """Client Groq pour un agent, avec secours automatiques (FALLBACK_<n>_*) et cassettes facultatives."""
    specs = _fallback_specs()
    if specs and not replay:
        # avec un secours : le principal n'insiste pas (1 nouvel essai, attente <= 5 s) ; c'est le secours qui prend le relais
        primary = GroqClient.for_agent(agent, model, max_retries=int(os.environ.get("PRIMARY_MAX_RETRIES", 1)),
                                       max_retry_wait=float(os.environ.get("PRIMARY_MAX_WAIT_S", 5)))
        client = FallbackClient([primary] + [build_fallback_client(agent, s, max_retries=2) for s in specs],
                                cooldown_s=float(os.environ.get("FALLBACK_COOLDOWN_S", 60)),
                                on_switch=lambda m: print(f"  [llm] {m}", flush=True))
    else:
        client = GroqClient.for_agent(agent, model)
    if cache_dir or replay:
        client = CachedClient(client, cache_dir or f"~/.cache/agentic-mlops/{agent}-cassettes", mode="replay" if replay else "auto")
    return client