"""Prepare a conservative retry after verification failure."""

from ..state import Agent1State


def retry(state: Agent1State) -> dict[str, object]:
    """Clear the failed plan while preserving the attempt counted by verify."""
    return {"improvement_plan": [], "changes": []}
