"""Prompt système du Judge (texte dans `system.md`)."""
from pathlib import Path

PROMPT = (Path(__file__).parent / "system.md").read_text(encoding="utf-8")
