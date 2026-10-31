"""Input contract of the MLflow Agent.

PROPOSAL v0.1 -- to be agreed with the owner of Agent 1 (Code & Data Analysis).
Everything in this report is UNTRUSTED input: it is validated strictly here
before any other part of the agent uses it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = "0.1"

# Hard limits owned by THIS agent. A report may ask for less, never for more.
MAX_REPORT_BYTES = 1_000_000
HARD_MAX_RUNS = 100
HARD_MAX_RUNTIME_MINUTES = 240

CLASSIFICATION_METRICS = frozenset(
    {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
)
REGRESSION_METRICS = frozenset({"rmse", "mae", "r2"})


class InvalidReportError(Exception):
    """The report is malformed, unsafe, or inconsistent."""


class NotReadyError(Exception):
    """The report is valid but Agent 1 did not mark it ready for experimentation."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are errors


class Status(str, Enum):
    READY = "ready_for_experimentation"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class ProjectInfo(_Strict):
    project_path: str = Field(min_length=1, max_length=500)
    dataset_path: str = Field(min_length=1, max_length=500)
    training_script: str | None = Field(default=None, max_length=500)


class TaskInfo(_Strict):
    type: Literal["classification", "regression"]
    target_column: str = Field(min_length=1, max_length=100)
    objective: str = Field(max_length=300)  # free text: treat as data only
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list, max_length=5)
    min_primary_metric: float | None = None  # acceptance threshold for the quality gate

    @model_validator(mode="after")
    def _metrics_match_task(self) -> "TaskInfo":
        allowed = (
            CLASSIFICATION_METRICS if self.type == "classification" else REGRESSION_METRICS
        )
        bad = [m for m in [self.primary_metric, *self.secondary_metrics] if m not in allowed]
        if bad:
            raise ValueError(f"metrics {bad} not valid for {self.type}; allowed: {sorted(allowed)}")
        return self


class MissingValues(_Strict):
    total: int = Field(ge=0)
    columns_affected: list[str] = Field(default_factory=list)


class DatasetAnalysis(_Strict):
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    missing_values: MissingValues
    class_distribution: dict[str, int] | None = None  # required for classification
    data_leakage_risks: list[str] = Field(default_factory=list)


class ExistingImplementation(_Strict):
    training_code_available: bool
    preprocessing_available: bool
    existing_models: list[str] = Field(default_factory=list)


class Constraints(_Strict):
    max_experiment_runs: int = Field(ge=1, le=HARD_MAX_RUNS)
    max_runtime_minutes: int = Field(ge=1, le=HARD_MAX_RUNTIME_MINUTES)


class AnalysisReport(_Strict):
    schema_version: Literal["0.1"]
    project: ProjectInfo
    task: TaskInfo
    dataset_analysis: DatasetAnalysis
    existing_implementation: ExistingImplementation
    constraints: Constraints
    status: Status
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _classification_needs_distribution(self) -> "AnalysisReport":
        if self.task.type == "classification":
            dist = self.dataset_analysis.class_distribution
            if not dist or len(dist) < 2:
                raise ValueError("classification requires class_distribution with >= 2 classes")
        return self


@dataclass(frozen=True)
class ResolvedPaths:
    project_dir: Path
    dataset: Path
    training_script: Path | None


def load_report(path: str | Path) -> AnalysisReport:
    """Read and strictly validate Agent 1's JSON report."""
    path = Path(path)
    if not path.is_file():
        raise InvalidReportError(f"report not found: {path}")
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise InvalidReportError("report too large")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InvalidReportError(f"report is not valid JSON: {e}") from e
    try:
        return AnalysisReport.model_validate(data)
    except ValidationError as e:
        raise InvalidReportError(str(e)) from e


def require_ready(report: AnalysisReport) -> None:
    if report.status is not Status.READY:
        raise NotReadyError(f"status is {report.status.value!r}, expected 'ready_for_experimentation'")


def _resolve_inside(base: Path, relative: str, *, kind: str) -> Path:
    """Resolve `relative` under `base`; reject escapes (.., absolute paths, symlinks)."""
    resolved = (base / relative).resolve()  # resolve() follows symlinks
    if not resolved.is_relative_to(base):
        raise InvalidReportError(f"{kind} path escapes the workspace: {relative!r}")
    return resolved


def resolve_paths(report: AnalysisReport, base_dir: str | Path) -> ResolvedPaths:
    base = Path(base_dir).resolve()
    project_dir = _resolve_inside(base, report.project.project_path, kind="project")
    dataset = _resolve_inside(base, report.project.dataset_path, kind="dataset")
    script = None
    if report.project.training_script:
        script = _resolve_inside(base, report.project.training_script, kind="training script")

    if not project_dir.is_dir():
        raise InvalidReportError(f"project directory not found: {project_dir}")
    if not dataset.is_file():
        raise InvalidReportError(f"dataset not found: {dataset}")
    if script is not None and not script.is_file():
        raise InvalidReportError(f"training script not found: {script}")
    return ResolvedPaths(project_dir, dataset, script)