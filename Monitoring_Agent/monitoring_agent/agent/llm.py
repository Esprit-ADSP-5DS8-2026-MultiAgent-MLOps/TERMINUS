"""
Client LLM : un seul protocole (chat/completions d'OpenAI) pour Groq, Gemini et Ollama.

Deux rôles, deux familles de modèles :
- décideur (Groq) : enquête avec des outils, diagnostique, propose une action ;
- juge (Gemini)   : relit et note. Un modèle d'une autre famille ne partage pas les mêmes biais.

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
    "mlx": ("http://127.0.0.1:8080/v1", None),      # mlx_lm.server (modèle local + adaptateur LoRA)
}
LOCAL = ("ollama", "mlx")
# Les catalogues changent vite (llama-3.3 et gemini-2.5-flash ont été retirés en 2026) :
# `python -m monitoring_agent llm-check` liste ce qui est disponible ; on surcharge dans .env.
DEFAULTS = {
    "MONITORING_DECIDER": "groq:openai/gpt-oss-120b",        # raisonnement + appels d'outils
    "MONITORING_DECIDER_FALLBACK": "groq:qwen/qwen3.8-27b",  # autre famille, quota séparé
    "MONITORING_JUDGE": "gemini:gemini-3.8-flash",           # autre fournisseur et autre famille que le décideur
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
    usage: dict = field(default_factory=dict)      # jetons : prompt_tokens, completion_tokens, total_tokens
    ms: float = 0.0                                # latence de l'appel


def usage_summary(responses: list[LLMResponse]) -> dict:
    """Agrège les appels d'une étape : nombre, jetons, latence, par modèle (pour la trace et Langfuse)."""
    out: dict = {"calls": len(responses), "prompt_tokens": 0, "completion_tokens": 0, "ms": 0.0, "models": {}}
    for r in responses:
        out["prompt_tokens"] += int(r.usage.get("prompt_tokens", 0) or 0)
        out["completion_tokens"] += int(r.usage.get("completion_tokens", 0) or 0)
        out["ms"] += r.ms
        out["models"][r.model] = out["models"].get(r.model, 0) + 1
    out["ms"] = round(out["ms"], 1)
    return out


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool | dict = False,
             temperature: float = 0.0) -> LLMResponse: ...
    # json_mode : False (texte libre), True (un objet JSON), ou un JSON Schema (sortie contrainte au décodage).


def _settings() -> dict[str, str]:
    env_file = Path(os.environ.get("MONITORING_ENV_FILE", PACKAGE_ROOT / ".env"))
    values = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.exists() else {}
    return {**values, **{k: v for k, v in os.environ.items() if k.startswith(("MONITORING_", "GROQ_", "GEMINI_", "LANGFUSE_"))}}


class OpenAICompatModel:
    def __init__(self, provider: str, model: str, api_key: str | None, base_url: str | None = None,
                 timeout_s: float = 90, max_tokens: int = 4096, retries: int = 2, max_wait_s: float = 20,
                 reasoning_effort: str | None = None):
        self.provider, self.model = provider, model
        self.name = f"{provider}:{model}"
        self.base_url = (base_url or PROVIDERS[provider][0]).rstrip("/")
        self.api_key, self.timeout_s, self.max_tokens = api_key, timeout_s, max_tokens
        self.retries, self.max_wait_s = retries, max_wait_s
        # Modèles « de raisonnement » (gpt-oss) : ils réfléchissent avant de répondre, et ces jetons
        # comptent dans max_tokens. « medium » : bon compromis qualité / latence / quota.
        # Gemini 3.x réfléchit longuement par défaut (juge : ~80 s) : « low » suffit pour noter une grille.
        self.reasoning_effort = reasoning_effort or ("medium" if "gpt-oss" in model else
                                                     "low" if provider == "gemini" else None)
        self.schema_ok = True   # passe à False si le fournisseur refuse response_format=json_schema

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        from .. import observability as obs
        params = {"temperature": temperature, "max_tokens": self.max_tokens, "tools": len(tools or []),
                  "response_format": "json_schema" if isinstance(json_mode, dict) else ("json" if json_mode else "text")}
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        with obs.observation(f"llm {self.name}", as_type="generation", model=self.model, input=messages,
                             model_parameters=params) as gen:
            resp = self._chat(messages, tools, json_mode, temperature)
            obs.update(gen, output=resp.raw_message or resp.text, metadata={"ms": resp.ms},
                       usage_details={"input": int(resp.usage.get("prompt_tokens", 0) or 0),
                                      "output": int(resp.usage.get("completion_tokens", 0) or 0)})
            return resp

    def _chat(self, messages, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                "max_tokens": self.max_tokens}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if isinstance(json_mode, dict) and self.schema_ok:
            # Décodage contraint : le modèle ne peut produire que des jetons conformes au schéma
            # (énumérations des causes / actions comprises). strict=False : « best effort » accepté par
            # Groq, Gemini et Ollama ; la validation Pydantic reste le contrôle final.
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": json_mode.get("title", "output"), "schema": json_mode,
                                                       "strict": False}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s)
            except httpx.HTTPError as exc:
                last, wait = f"réseau : {exc}", 2.0 * (attempt + 1)
            else:
                if r.status_code == 200:
                    resp = self._parse(r.json())
                    resp.ms = round((time.perf_counter() - t0) * 1000, 1)
                    return resp
                last = f"HTTP {r.status_code} : {r.text[:700]}"
                if r.status_code == 400 and "json_validate_failed" in r.text:
                    # Groq a refusé une sortie non conforme : on rend le brouillon, `structured` le fera corriger.
                    return LLMResponse(text=_failed_generation(r), model=self.name,
                                       ms=round((time.perf_counter() - t0) * 1000, 1))
                if r.status_code == 400 and isinstance(json_mode, dict) and self.schema_ok and _schema_refused(r.text):
                    self.schema_ok = False                    # fournisseur sans json_schema : repli json_object
                    return self._chat(messages, tools, True, temperature)
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
        usage = {k: v for k, v in (data.get("usage") or {}).items()
                 if k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls, model=self.name, raw_message=clean,
                           usage=usage)


def _failed_generation(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error", {}).get("failed_generation") or "")
    except ValueError:
        return ""


def _schema_refused(text: str) -> bool:
    t = text.lower()
    return "response_format" in t or "json_schema" in t or "responseschema" in t


def llm_schema(model: type[BaseModel]) -> dict:
    """Schéma Pydantic -> JSON Schema portable : références internes déroulées, titres/défauts retirés."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def clean(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return clean(defs[node["$ref"].split("/")[-1]])
            out = {k: clean(v) for k, v in node.items() if k not in ("title", "default")}
            if "properties" in out and "additionalProperties" not in out:
                out["required"] = list(out["properties"])   # côté LLM, tous les champs sont attendus
            return out
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    out = clean(raw)
    out["title"] = model.__name__
    return out


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


def build_model(spec: str, settings: dict[str, str], timeout_s: float = 90,
                retries: int = 2) -> OpenAICompatModel | None:
    """« groq:openai/gpt-oss-120b » -> client, ou None si la clé manque."""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise LLMError(f"modèle mal défini : {spec!r} (attendu fournisseur:modèle, fournisseurs {list(PROVIDERS)})")
    key_env = PROVIDERS[provider][1]
    key = settings.get(key_env) if key_env else None
    if key_env and not key:
        return None
    return OpenAICompatModel(provider, model, key, settings.get(f"MONITORING_{provider.upper()}_BASE_URL"),
                             timeout_s=timeout_s, retries=retries,
                             reasoning_effort=settings.get(f"MONITORING_{provider.upper()}_REASONING_EFFORT")
                             or (settings.get("MONITORING_REASONING_EFFORT") if provider == "groq" else None))


def _chain(specs: list[str], settings: dict[str, str]) -> ChatModel | None:
    models = [m for m in (build_model(x, settings) for x in specs if x) if m is not None]
    return FallbackModel(models) if len(models) > 1 else (models[0] if models else None)


def decider_and_judge() -> tuple[ChatModel | None, ChatModel | None]:
    """Décideur (avec repli) et juge, depuis .env.

    Si le décideur est local (mlx / ollama : le Qwen fine-tuné), il ne fait que diagnostiquer, sur les faits
    seuls (le format appris à l'entraînement) ; l'enquête avec outils reste à un grand modèle
    (MONITORING_INVESTIGATOR). Repli automatique sur Groq si le serveur local est arrêté."""
    s = _settings()
    spec = s.get("MONITORING_DECIDER", DEFAULTS["MONITORING_DECIDER"])
    fallback = s.get("MONITORING_DECIDER_FALLBACK", DEFAULTS["MONITORING_DECIDER_FALLBACK"])
    decider = _chain([spec, fallback], s)
    if decider is not None and spec.partition(":")[0] in LOCAL:
        decider.local = True
        decider.investigator = _chain([s.get("MONITORING_INVESTIGATOR", DEFAULTS["MONITORING_DECIDER"]), fallback], s)
    # Le juge n'est qu'un avis en plus : s'il ne répond pas assez vite, la règle R7 passe au second avis
    # des règles. Avec 90 s x 3 essais on attendait jusqu'à 4 minutes (vu en démo) : un seul essai de 60 s.
    judge = build_model(s.get("MONITORING_JUDGE", DEFAULTS["MONITORING_JUDGE"]), s,
                        timeout_s=float(s.get("MONITORING_JUDGE_TIMEOUT_S", 60)), retries=0)
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
               attempts: int = 2, meter: list | None = None) -> T:
    """`meter` : liste qui reçoit chaque réponse brute (jetons, latence) pour la trace."""
    msgs = list(messages)
    last_error = ""
    json_schema = llm_schema(schema)
    for _ in range(attempts):
        resp = model.chat(msgs, json_mode=json_schema, temperature=temperature)
        if meter is not None:
            meter.append(resp)
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
