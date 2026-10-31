import pytest

from mlflow_agent.config import DEFAULT_LLM, ConfigError, resolve_llm


def cfg(tmp_path, text):
    f = tmp_path / "c.toml"
    f.write_text(text)
    return f


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_AGENT_LLM", raising=False)


def test_command_line_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_AGENT_LLM", "env-model")
    assert resolve_llm("cli-model", cfg(tmp_path, 'llm = "file-model"')) == ("cli-model", "command line")


def test_env_beats_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_AGENT_LLM", "env-model")
    spec, source = resolve_llm(None, cfg(tmp_path, 'llm = "file-model"'))
    assert spec == "env-model" and "environment" in source


def test_file_is_used(tmp_path):
    assert resolve_llm(None, cfg(tmp_path, 'llm = "gemini:gemini-3.5-flash-lite"')) == (
        "gemini:gemini-3.5-flash-lite", "config file")


def test_default_when_nothing_set(tmp_path):
    assert resolve_llm(None, tmp_path / "missing.toml") == (DEFAULT_LLM, "built-in default")


@pytest.mark.parametrize("bad", ['llm = "x; rm -rf /"', "llm = 5", 'llm = "a b"'])
def test_invalid_setting_rejected(tmp_path, bad):
    with pytest.raises(ConfigError):
        resolve_llm(None, cfg(tmp_path, bad))


def test_broken_toml_rejected(tmp_path):
    with pytest.raises(ConfigError):
        resolve_llm(None, cfg(tmp_path, "llm = = ="))
