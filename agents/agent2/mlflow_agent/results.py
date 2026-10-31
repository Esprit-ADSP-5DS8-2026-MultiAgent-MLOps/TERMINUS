"""Output contract of the MLflow Agent (PROPOSAL v0.1, to agree with Agent 3's owner)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mlflow_agent.plans import ExperimentPlan


class QualityGate(BaseModel):
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float | None] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    schema_version: str = "0.1"
    status: Literal["success", "failed", "no_eligible_model"]
    primary_metric: str
    experiment_id: str | None = None
    run_id: str | None = None
    plan: ExperimentPlan | None = None
    best_model_name: str | None = None
    best_params: dict[str, Any] = Field(default_factory=dict)
    train_score: float | None = None
    validation_score: float | None = None
    test_score: float | None = None
    baseline_test_score: float | None = None
    model_uri: str | None = None
    model_version: str | None = None
    quality_gate: QualityGate | None = None
    trials_completed: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
