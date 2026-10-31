import sys
from pathlib import Path

import pytest

# agents/agent2/ (le paquet mlflow_agent) et la racine du dépôt (common/, orchestrator/) importables
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2])]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Jamais de vrai LLM ni de vraie base MLflow de la machine dans les tests."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_AGENT_LLM", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("AGENT2_MLFLOW_ARTIFACTS", str(tmp_path / "mlartifacts"))
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
