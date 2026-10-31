"""Restore the untouched project snapshot before data analysis."""

from ..state import Agent1State
from .quality_analysis import quality_analysis
from .verify import restore_snapshot


def revert_and_continue(state: Agent1State) -> dict[str, object]:
    """Restore the workspace and continue without further code changes."""
    try:
        restore_snapshot(state.get("workspace_path", ""), state.get("snapshot_path", ""))
        quality = quality_analysis(state)
        return {
            "changes": [],
            "verification_passed": False,
            "verification_details": {
                "layer": 5,
                "name": "revert_and_continue",
                "message": "Verification failed after the maximum retry count; snapshot restored.",
            },
            "quality_after": quality.get("quality_before", {}),
            "quality_issues": quality.get("quality_issues", []),
            "tool_errors": [*state.get("tool_errors", []), *quality.get("tool_errors", [])],
        }
    except Exception as exc:
        return {"errors": [{"node": "revert_and_continue", "message": str(exc)}]}
