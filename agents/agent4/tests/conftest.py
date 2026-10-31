import sys
from pathlib import Path

import pytest

# agents/agent4/ (le package) et la racine du dépôt (common/) importables sans installation
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2])]


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch, tmp_path):
    """Les tests n'appellent jamais un vrai LLM, même si .env (agents/agent4/.env ou racine du dépôt) contient des clés."""
    monkeypatch.setenv("MONITORING_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("MONITORING_MEMORY", "0")          # pas la mémoire réelle de la machine
    monkeypatch.setenv("MONITORING_EMBEDDER", "hash")     # jamais de téléchargement de modèle
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                "MONITORING_NOTIFY_WEBHOOK", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL",
                "SLACK_APPROVERS", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
