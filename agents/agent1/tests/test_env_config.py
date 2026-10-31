"""Tests for lazy package-local environment loading."""

from pathlib import Path

from agents.agent1 import config
from agents.agent1.runner import run_agent1


def _use_env_file(monkeypatch: object, tmp_path: Path, content: str) -> None:
    """Point the lazy loader at a temporary package marker file."""
    marker = tmp_path / "config.py"
    marker.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", str(marker))
    monkeypatch.setattr(config, "_env_loaded", False)
    (tmp_path / ".env").write_text(content, encoding="utf-8")


def test_values_from_dotenv_are_loaded_lazily(tmp_path: Path, monkeypatch: object) -> None:
    """Quoted values from the package-local file are used on first config access."""
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ML_LLM_MODEL", raising=False)
    _use_env_file(monkeypatch, tmp_path, 'XAI_API_KEY="dotenv-secret"\nML_LLM_MODEL=dotenv-model\n')
    result = config.get_ml_llm_config()
    assert result["api_key"] == "dotenv-secret"
    assert result["model"] == "dotenv-model"


def test_real_environment_wins_over_dotenv(tmp_path: Path, monkeypatch: object) -> None:
    """Already-set process variables are never overridden."""
    monkeypatch.setenv("XAI_API_KEY", "real-secret")
    monkeypatch.setenv("ML_LLM_MODEL", "real-model")
    _use_env_file(monkeypatch, tmp_path, "XAI_API_KEY=dotenv-secret\nML_LLM_MODEL=dotenv-model\n")
    result = config.get_ml_llm_config()
    assert result["api_key"] == "real-secret"
    assert result["model"] == "real-model"


def test_missing_dotenv_is_fine(tmp_path: Path, monkeypatch: object) -> None:
    """A missing package-local file leaves normal defaults available."""
    marker = tmp_path / "config.py"
    marker.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", str(marker))
    monkeypatch.setattr(config, "_env_loaded", False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert config.get_ml_llm_config()["api_key"] == ""


def test_ml_dotenv_strips_whitespace_and_uses_generic_key(tmp_path: Path, monkeypatch: object) -> None:
    """Whitespace around ML values is removed and ML_LLM_API_KEY is preferred."""
    monkeypatch.delenv("ML_LLM_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ML_LLM_MODEL", raising=False)
    monkeypatch.delenv("ML_LLM_BASE_URL", raising=False)
    _use_env_file(
        monkeypatch,
        tmp_path,
        "ML_LLM_API_KEY= \t ml-secret \t\nML_LLM_MODEL= \t model-x \t\nML_LLM_BASE_URL= \thttps://example.test/v1/ \t\n",
    )
    result = config.get_ml_llm_config()
    assert result["api_key"] == "ml-secret"
    assert result["model"] == "model-x"
    assert result["base_url"] == "https://example.test/v1"


def test_secret_value_never_appears_in_agent_output(tmp_path: Path, monkeypatch: object) -> None:
    """The configured secret is not serialized by the public agent output."""
    secret = "dotenv-secret-value"
    monkeypatch.setenv("XAI_API_KEY", secret)
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    csv_path = project / "data.csv"
    csv_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    assert secret not in str(run_agent1(str(project), str(csv_path)))
