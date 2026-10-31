"""Build a deterministic, bounded plan for local function edits."""

from typing import Any

from ..config import TIER3_ENABLED, get_llm_config
from ..llm import availability_details, llm_available
from ..state import Agent1State


def _plan_item(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge custom function issues into one edit item."""
    codes = {str(issue.get("code")) for issue in issues}
    if {"DOC001", "ANN001"} <= codes:
        kind = "missing_both"
    elif "DOC001" in codes:
        kind = "missing_docstring"
    else:
        kind = "missing_type_hints"
    first = issues[0]
    return {
        "file": first.get("file", ""),
        "function": first.get("function", ""),
        "line_range": [
            first.get("line", 1),
            max(int(issue.get("end_lineno", first.get("line", 1))) for issue in issues),
        ],
        "kind": kind,
        "tier": 2,
    }


def plan_improvements(state: Agent1State) -> dict[str, object]:
    """Create a deterministic plan without calling the LLM for planning."""
    if 2 not in state.get("allowed_tiers", []):
        return {"improvement_plan": [], "llm": {"planned": 0}, "llm_note": "Tier 2 is not allowed."}
    config = get_llm_config()
    if not llm_available():
        details = availability_details()
        note = details["note"] or "LLM endpoint unavailable; skipped function edits."
        return {
            "improvement_plan": [],
            "llm": {
                "planned": 0,
                "used": False,
                "model": config["model_name"],
                "endpoint_ok": False,
                "endpoint_error": details["endpoint_error"],
                "applied": 0,
                "rejected": [],
                "note": note,
            },
            "llm_note": note,
        }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for issue in state.get("quality_issues", []):
        if issue.get("code") not in {"DOC001", "ANN001"} or not issue.get("function"):
            continue
        key = (str(issue.get("file")), str(issue.get("function")))
        grouped.setdefault(key, []).append(issue)
    plan = []
    for issues in grouped.values():
        item = _plan_item(issues)
        if int(item["line_range"][1]) - int(item["line_range"][0]) + 1 > 45:
            continue
        if state.get("attempts", 0) > 0:
            if item["kind"] == "missing_both":
                item["kind"] = "missing_docstring"
            elif item["kind"] == "missing_type_hints":
                continue
        plan.append(item)
    limit = min(int(config["max_llm_items"]), len(plan))
    if state.get("attempts", 0) > 0:
        limit //= 2
    plan = plan[:limit]
    note = "" if plan else "No eligible deterministic docstring/type-hint edits."
    return {
        "improvement_plan": plan,
        "llm": {
            "planned": len(plan),
            "used": False,
            "model": config["model_name"],
            "endpoint_ok": True,
            "applied": 0,
            "rejected": [],
            "note": note,
        },
        "llm_note": note,
        "tier3_enabled": TIER3_ENABLED,
    }
