"""State schema owned by Agent 1."""

from typing import Any, TypedDict


class Agent1State(TypedDict):
    """State passed between Agent 1 graph nodes."""

    project_path: str
    workspace_path: str
    snapshot_path: str
    csv_path: str
    errors: list[dict[str, str]]
    baseline: dict[str, Any] | None
    baseline_ok: bool
    allowed_tiers: list[int]
    quality_issues: list[dict[str, Any]]
    improvement_plan: list[dict[str, Any]]
    changes: list[dict[str, Any]]
    verification_passed: bool | None
    verification_details: dict[str, Any] | None
    attempts: int
    quality_before: dict[str, Any]
    quality_after: dict[str, Any]
    profile: dict[str, Any]
    problem_type: str | None
    subtype: str | None
    target: str | None
    features: dict[str, list[str]]
    confidence: float
    tool_errors: list[dict[str, Any]]
    output: dict[str, Any]
