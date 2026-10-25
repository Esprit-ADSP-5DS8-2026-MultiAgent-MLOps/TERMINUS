import sys
from pathlib import Path

import pytest

# Monitoring_Agent/ (le package) et la racine du repo (common/) importables sans installation
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch, tmp_path):
    """Les tests n'appellent JAMAIS un vrai LLM, même si Monitoring_Agent/.env contient des clés."""
    monkeypatch.setenv("MONITORING_ENV_FILE", str(tmp_path / "absent.env"))
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
