"""Experiment plan proposed by the LLM planner.

The LLM output is UNTRUSTED. It is parsed and checked here against a strict
schema and against the validated analysis report before anything runs.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mlflow_agent.schemas import AnalysisReport

MAX_PLAN_BYTES = 20_000

ClassificationModel = Literal["logistic_regression", "random_forest", "gradient_boosting"]
RegressionModel = Literal["ridge", "random_forest_regressor", "gradient_boosting_regressor"]

CLASSIFICATION_MODELS = frozenset({"logistic_regression", "random_forest", "gradient_boosting"})
REGRESSION_MODELS = frozenset({"ridge", "random_forest_regressor", "gradient_boosting_regressor"})


class InvalidPlanError(Exception):
    """The plan is malformed or violates the report's constraints."""


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ClassificationModel | RegressionModel] = Field(min_length=1, max_length=3)
    primary_metric: str
    search_method: Literal["optuna"] = "optuna"
    max_trials: int = Field(ge=1, le=100)
    rationale: str = Field(max_length=500)  # explanation only, never executed


def parse_plan(text: str) -> ExperimentPlan:
    """Turn the LLM's raw text into a validated plan."""
    if len(text.encode("utf-8")) > MAX_PLAN_BYTES:
        raise InvalidPlanError("plan too large")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidPlanError(f"plan is not valid JSON: {e}") from e
    try:
        return ExperimentPlan.model_validate(data)
    except ValidationError as e:
        raise InvalidPlanError(str(e)) from e


def check_plan_against_report(plan: ExperimentPlan, report: AnalysisReport) -> None:
    """Deterministic rules the LLM cannot override."""
    allowed = CLASSIFICATION_MODELS if report.task.type == "classification" else REGRESSION_MODELS
    bad = [m for m in plan.models if m not in allowed]
    if bad:
        raise InvalidPlanError(f"models {bad} not valid for {report.task.type}")
    if len(set(plan.models)) != len(plan.models):
        raise InvalidPlanError("duplicate models in plan")
    if plan.primary_metric != report.task.primary_metric:
        raise InvalidPlanError(
            f"plan metric {plan.primary_metric!r} != required {report.task.primary_metric!r}"
        )
    if plan.max_trials > report.constraints.max_experiment_runs:
        raise InvalidPlanError("plan exceeds the allowed number of experiment runs")