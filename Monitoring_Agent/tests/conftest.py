import sys
from pathlib import Path

import pytest

# Monitoring_Agent/ (le package) et la racine du repo (common/) importables sans installation
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch, tmp_path):
    """Les tests n'appellent jamais un vrai LLM, même si Monitoring_Agent/.env contient des clés."""
    monkeypatch.setenv("MONITORING_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("MONITORING_MEMORY", "0")          # pas la mémoire réelle de la machine
    monkeypatch.setenv("MONITORING_EMBEDDER", "hash")     # jamais de téléchargement de modèle
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                "MONITORING_NOTIFY_WEBHOOK", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL",
                "SLACK_APPROVERS"):
        monkeypatch.delenv(key, raising=False)
