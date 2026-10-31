"""Adapters between Agent 1 state and a future parent graph state."""

from typing import Any, Mapping

from .state import Agent1State


def from_main_state(main_state: Mapping[str, Any]) -> Agent1State:
    """Build Agent 1 state from a parent graph state placeholder."""
    return dict(main_state)  # type: ignore[return-value]


def to_main_state(state: Agent1State) -> dict[str, Any]:
    """Convert Agent 1 state to a parent graph state placeholder."""
    return dict(state)
