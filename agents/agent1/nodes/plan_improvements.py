"""Plan-improvements node stub."""

from ..state import Agent1State


def plan_improvements(state: Agent1State) -> dict[str, list[dict[str, object]]]:
    """Print the node name and return an empty improvement plan."""
    print("plan_improvements")
    return {"improvement_plan": []}
