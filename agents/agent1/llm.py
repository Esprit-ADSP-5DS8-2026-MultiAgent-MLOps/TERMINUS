"""Small OpenAI-compatible Groq wrapper used only for function edits."""

import json
import re
import urllib.error
import urllib.request
from typing import Any

from .config import get_llm_config
from .config import get_ml_llm_config

SYSTEM = (
    "You are a careful Python code improver. You receive one function and an issue. "
    "Return ONLY the improved function as plain Python code, with no explanation and "
    "no markdown. Never change the behavior or logic of the function."
)
USER = "Issue: {issue}\n\nFunction:\n{function_source}"
ISSUES = {
    "missing_docstring": "The function has no docstring. Add a concise, accurate docstring.",
    "missing_type_hints": "The function has no type hints. Add type hints for the parameters and the return value.",
    "missing_both": "The function has no docstring and no type hints. Add both.",
}

_availability: bool | None = None
_endpoint_error: str | None = None
_availability_note = ""
_ml_error = ""


def complete_text(system: str, user: str) -> str:
    """POST one chat completion request and return content, or an empty string."""
    config = get_llm_config()
    payload = json.dumps(
        {
            "model": config["model_name"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 800,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f'{config["base_url"].rstrip("/")}/chat/completions',
        data=payload,
        headers={"Authorization": f'Bearer {config["api_key"]}', "Content-Type": "application/json"},
        method="POST",
    )
    request.add_header("User-Agent", "agent1/1.0")
    try:
        with urllib.request.urlopen(request, timeout=config["llm_timeout"]) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])
    except Exception:
        return ""


def llm_available() -> bool:
    """Cache whether the configured Groq endpoint responds to its models request."""
    global _availability, _endpoint_error, _availability_note
    config = get_llm_config()
    if not config["enable_llm"]:
        _endpoint_error = None
        _availability_note = "LLM disabled by configuration."
        return False
    if _availability is not None:
        return _availability
    request = urllib.request.Request(
        f'{config["base_url"].rstrip("/")}/models',
        # Same User-Agent as the other calls: Groq (Cloudflare) answers 403 to the default "Python-urllib/x.y".
        headers={"User-Agent": "agent1/1.0",
                 **({"Authorization": f'Bearer {config["api_key"]}'} if config["api_key"] else {})},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(int(config["llm_timeout"]), 3)) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
        if status == 200:
            _availability = True
            _endpoint_error = None
            _availability_note = ""
            return True
        _endpoint_error = f"HTTP status {status}"
        _availability_note = _endpoint_error
    except urllib.error.HTTPError as exc:
        _endpoint_error = f"HTTP status {exc.code}: {exc.reason}"
        _availability_note = _endpoint_error
    except Exception as exc:
        _endpoint_error = str(exc)
        _availability_note = _endpoint_error
    return False


def availability_details() -> dict[str, str | None]:
    """Return the latest endpoint note and real error text."""
    return {"note": _availability_note, "endpoint_error": _endpoint_error}


def reset_availability_cache() -> None:
    """Reset endpoint availability for tests."""
    global _availability, _endpoint_error, _availability_note
    _availability = None
    _endpoint_error = None
    _availability_note = ""


class FakeLLM:
    """Scripted no-network client for unit tests."""

    def __init__(self, outputs: list[str]) -> None:
        """Create a fake client with sequential completion outputs."""
        self.outputs = list(outputs)

    def complete_text(self, system: str, user: str) -> str:
        """Return the next scripted completion."""
        return self.outputs.pop(0) if self.outputs else ""


class LLMClient:
    """Provider-facing client that delegates to the module wrapper."""

    def complete_text(self, system: str, user: str) -> str:
        """Complete one edit prompt."""
        return complete_text(system, user)


def get_client() -> LLMClient:
    """Return the configured provider client."""
    return LLMClient()


def complete_ml(system: str, user: str) -> str:
    """Request one ML review function replacement without exposing the API key."""
    global _ml_error
    _ml_error = ""
    config = get_ml_llm_config()
    if not config["api_key"]:
        _ml_error = "API key is not configured"
        return ""
    token_budget = int(config["max_tokens"])
    for attempt in range(2):
        payload_data = {"model": config["model"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": token_budget}
        if config["reasoning_effort"] in {"low", "medium", "high"}:
            payload_data["reasoning_effort"] = config["reasoning_effort"]
        payload = json.dumps(payload_data).encode("utf-8")
        request = urllib.request.Request(f'{config["base_url"]}/chat/completions', data=payload, headers={"Authorization": f'Bearer {config["api_key"]}', "Content-Type": "application/json", "User-Agent": "agent1/1.0"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=config["timeout"]) as response:
                body = json.loads(response.read().decode("utf-8"))
            choice = body["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            if content:
                return str(content)
            finish_reason = choice.get("finish_reason", "unknown")
            if attempt == 1:
                _ml_error = f"empty answer, finish_reason={finish_reason}"
                return ""
            token_budget *= 2
        except urllib.error.HTTPError as exc:
            _ml_error = f"HTTP status {exc.code}: {exc.reason}"
            return ""
        except Exception as exc:
            _ml_error = str(exc).replace(str(config["api_key"]), "[redacted]")
            return ""
    return ""


def strip_json_fences(content: str) -> str:
    """Remove optional markdown fences from a JSON completion."""
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else content).strip()


def complete_ml_json(system: str, user: str) -> dict[str, Any] | None:
    """Parse one JSON completion and retry once when the answer is invalid."""
    global _ml_error
    for _ in range(2):
        content = complete_ml(system, user)
        if not content:
            return None
        try:
            value = json.loads(strip_json_fences(content))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            _ml_error = "invalid JSON answer"
    return None


def ml_error_details() -> str:
    """Return the latest sanitized ML completion error."""
    return _ml_error
