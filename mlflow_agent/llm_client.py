"""Minimal LLM clients. The planner only depends on the `complete` method."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol


class LLMError(Exception):
    """The LLM call failed (network, timeout, bad response)."""


class QuotaError(LLMError):
    """The provider says the quota or rate limit is used up (HTTP 429)."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OllamaClient:
    """Talks to a local Ollama server (default http://localhost:11434)."""

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 120):
        if not host.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError("only a local Ollama host is allowed")
        self.name = f"ollama:{model}"
        self._model, self._host, self._timeout = model, host, timeout

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"ollama call failed: {e}") from e


class FakeClient:
    """Returns scripted answers. Used in tests, so tests never need a real LLM."""

    def __init__(self, answers: list[str]):
        self.name = "fake"
        self._answers = list(answers)

    def complete(self, system: str, user: str) -> str:
        if not self._answers:
            raise LLMError("no scripted answers left")
        return self._answers.pop(0)


class GeminiClient:
    """Google Gemini through its REST API. The key is read from an environment variable,
    sent in a header (never in the URL), and redacted from error messages."""

    HOST = "https://generativelanguage.googleapis.com"

    def __init__(self, model: str, api_key_env: str = "GEMINI_API_KEY", timeout: int = 60):
        model = model.removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", model):
            raise ValueError("invalid Gemini model name")
        key = os.environ.get(api_key_env)
        if not key:
            raise LLMError(f"environment variable {api_key_env} is not set")
        self.name = f"gemini:{model}"
        self._model, self._key, self._timeout = model, key, timeout

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }).encode()
        url = f"{self.HOST}/v1beta/models/{urllib.parse.quote(self._model, safe='')}:generateContent"
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "x-goog-api-key": self._key})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 2:
                    time.sleep(2 ** (attempt + 1))  # rate limit or overload: wait, retry
                    continue
                detail = e.read()[:300].decode("utf-8", "replace").replace(self._key, "***")
                if e.code == 429:
                    raise QuotaError(f"gemini HTTP 429 (quota): {detail}") from None
                raise LLMError(f"gemini HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
                    json.JSONDecodeError) as e:
                raise LLMError(f"gemini call failed: {type(e).__name__}") from None
        raise LLMError("gemini: retries exhausted")


def make_client(spec: str) -> LLMClient:
    """'gemini:<model>' -> Gemini, 'ollama:<model>' or a bare name -> local Ollama."""
    if spec.startswith("gemini:"):
        models = [m for m in spec.split(":", 1)[1].split(",") if m]
        return GeminiClient(models[0]) if len(models) == 1 else GeminiPool(models)
    if spec.startswith("ollama:"):
        return OllamaClient(spec.split(":", 1)[1])
    return OllamaClient(spec)


class GeminiPool:
    """Tries several Gemini models in order, so one used-up quota does not stop the agent.

    quota (429)    -> skip that model for COOLDOWN_S seconds, try the next one
    not found (404)-> skip that model for the rest of the run
    other errors   -> try the next model for this call only
    """

    COOLDOWN_S = 300

    def __init__(self, models: list[str], timeout: int = 60):
        if not models:
            raise ValueError("no Gemini models given")
        self._clients = [GeminiClient(m, timeout=timeout) for m in models]
        self._blocked_until: dict[str, float] = {}
        self.name = "gemini:" + ",".join(c._model for c in self._clients)
        self.last_model: str | None = None

    def complete(self, system: str, user: str) -> str:
        problems: list[str] = []
        for c in self._clients:
            if self._blocked_until.get(c._model, 0.0) > time.monotonic():
                continue
            try:
                out = c.complete(system, user)
                self.last_model = c._model
                return out
            except QuotaError:
                self._blocked_until[c._model] = time.monotonic() + self.COOLDOWN_S
                problems.append(f"{c._model}: quota")
            except LLMError as e:
                if "HTTP 404" in str(e):
                    self._blocked_until[c._model] = float("inf")
                problems.append(f"{c._model}: {str(e)[:60]}")
        raise LLMError("no Gemini model available (" + "; ".join(problems or ["all cooling down"]) + ")")
