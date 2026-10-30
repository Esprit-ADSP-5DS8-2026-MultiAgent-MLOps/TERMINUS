"""Choose the LLM: --model flag > $MLFLOW_AGENT_LLM > mlflow_agent.toml > default.

The config file holds no secrets. The Gemini key comes only from the environment.
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

DEFAULT_LLM = "qwen2.5:3b"
CONFIG_FILE = "mlflow_agent.toml"
_SPEC = re.compile(r"[A-Za-z0-9._:/,-]{1,300}")


class ConfigError(Exception):
    """The LLM setting is missing or malformed."""


def resolve_llm(cli_value: str | None = None,
                config_path: str | Path = CONFIG_FILE) -> tuple[str, str]:
    """Return (llm spec, where it came from)."""
    if cli_value:
        spec, source = cli_value, "command line"
    elif os.environ.get("MLFLOW_AGENT_LLM"):
        spec, source = os.environ["MLFLOW_AGENT_LLM"], "environment variable MLFLOW_AGENT_LLM"
    elif Path(config_path).is_file():
        try:
            data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            raise ConfigError(f"{config_path} is not valid TOML: {e}") from e
        spec, source = data.get("llm", DEFAULT_LLM), "config file"
    else:
        spec, source = DEFAULT_LLM, "built-in default"
    if not isinstance(spec, str) or not _SPEC.fullmatch(spec):
        raise ConfigError(f"invalid llm setting {spec!r} (from {source})")
    return spec, source
