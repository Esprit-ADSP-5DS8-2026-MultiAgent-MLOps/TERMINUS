"""LLM planner: ask for a plan, validate it, retry once, else fall back safely."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mlflow_agent.llm_client import LLMClient, LLMError
from mlflow_agent.plans import (
    ExperimentPlan, InvalidPlanError, check_plan_against_report, parse_plan,
)
from mlflow_agent.prompts import SYSTEM_PROMPT, build_user_prompt
from mlflow_agent.schemas import AnalysisReport

MAX_ATTEMPTS = 2


@dataclass
class PlanResult:
    plan: ExperimentPlan
    source: Literal["llm", "fallback"]
    attempts: int
    errors: list[str] = field(default_factory=list)


def _strip_fences(text: str) -> str:
    """Models often wrap JSON in ```json fences. Remove them before strict parsing."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rstrip()
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def default_plan(report: AnalysisReport) -> ExperimentPlan:
    """Deterministic safe plan, used when the LLM fails."""
    if report.task.type == "classification":
        models = ["logistic_regression", "random_forest"]
    else:
        models = ["ridge", "random_forest_regressor"]
    return ExperimentPlan(
        models=models,
        primary_metric=report.task.primary_metric,
        search_method="optuna",
        max_trials=min(20, report.constraints.max_experiment_runs),
        rationale="Default fallback plan (LLM plan unavailable or invalid).",
    )


def plan_experiment(report: AnalysisReport, client: LLMClient) -> PlanResult:
    user_prompt = build_user_prompt(report)
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = client.complete(SYSTEM_PROMPT, user_prompt)
            plan = parse_plan(_strip_fences(raw))
            check_plan_against_report(plan, report)
            return PlanResult(plan, "llm", attempt, errors)
        except (LLMError, InvalidPlanError) as e:
            errors.append(f"attempt {attempt}: {str(e)[:200]}")
    return PlanResult(default_plan(report), "fallback", MAX_ATTEMPTS, errors)
