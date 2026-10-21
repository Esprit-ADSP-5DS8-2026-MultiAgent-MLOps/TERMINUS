"""Swappable LLM wrapper for Agent 1."""

from typing import Any


class LLMClient:
    """Placeholder client that keeps provider details behind one interface."""

    def __init__(self, model: str) -> None:
        """Create a stub client for the configured model."""
        self.model = model

    def plan_improvements(
        self, issues: list[dict[str, Any]], allowed_tiers: list[int]
    ) -> list[dict[str, Any]]:
        """Return an empty plan until an LLM provider is integrated."""
        return []

    def resolve_data_ambiguity(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Return no opinion until an LLM provider is integrated."""
        return {}
