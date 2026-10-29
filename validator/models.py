"""Schémas Pydantic : rapport de validation (sortie vers l'orchestrateur) et manifest (entrée)."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    BLOCKER = "BLOCKER"      # rejette le projet
    CONFIRM = "CONFIRM"      # demande une confirmation humaine
    WARNING = "WARNING"      # accepté mais à signaler
    INFO = "INFO"


class Verdict(str, Enum):
    ACCEPTED = "ACCEPTED"
    ACCEPTED_WITH_WARNINGS = "ACCEPTED_WITH_WARNINGS"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    REJECTED = "REJECTED"


class Finding(BaseModel):
    rule: str                      # C1..C12
    code: str                      # identifiant stable, ex. C6.UNPINNED
    severity: Severity
    level: str                     # structure | manifest | static | data | dynamic
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    hint: Optional[str] = None     # piste de correction (exploitable par un agent)
    evidence: Optional[str] = None


class LevelInfo(BaseModel):
    name: str
    status: Literal["done", "skipped"]
    reason: Optional[str] = None
    duration_s: float = 0.0


REPORT_SCHEMA_VERSION = "1.0"   # version du FORMAT du rapport (distincte de contract_version)


class Confirmation(BaseModel):
    confirmed_by: str
    confirmed_at: str
    note: Optional[str] = None
    manifest_sha256: Optional[str] = None


class CarryForward(BaseModel):
    code: str
    message: str
    source: Literal["finding", "derived"]


class Handoff(BaseModel):
    """Ce dont l'agent suivant a besoin, sans parcourir la liste des constats."""
    status: Literal["READY", "READY_WITH_WARNINGS", "AWAITING_CONFIRMATION", "INCOMPLETE", "BLOCKED"]
    can_proceed: bool
    requires_human_confirmation: bool
    reason: Optional[str] = None
    task: Optional[str] = None
    target: Optional[str] = None
    primary_metric: Optional[str] = None
    min_threshold: Optional[float] = None
    data_path: Optional[str] = None
    data_summary: dict[str, Any] = Field(default_factory=dict)
    feature_columns: list[str] = Field(default_factory=list)
    entrypoints: dict[str, str] = Field(default_factory=dict)
    interfaces: dict[str, str] = Field(default_factory=dict)
    model_artifact: str = "model.joblib"
    pinned_dependencies: dict[str, str] = Field(default_factory=dict)
    observed_metrics: Optional[dict[str, float]] = None
    deterministic: Optional[bool] = None
    carry_forward: list[CarryForward] = Field(default_factory=list)


class ValidationReport(BaseModel):
    contract_version: str
    project: str
    verdict: Verdict
    summary: dict[str, int]
    findings: list[Finding]
    levels: list[LevelInfo]
    facts: dict[str, Any] = Field(default_factory=dict)
    observed_metrics: Optional[dict[str, Any]] = None
    sandbox: Optional[dict[str, Any]] = None
    duration_s: float = 0.0
    # --- passage de relais (renseignés par validator/handoff.py) ---
    schema_version: str = REPORT_SCHEMA_VERSION
    run_id: str = ""
    created_at: str = ""
    project_hash: Optional[str] = None
    hashed_files: int = 0
    snapshot_path: Optional[str] = None
    confirmation: Optional[Confirmation] = None
    handoff: Optional[Handoff] = None
    report_sha256: Optional[str] = None


# ---- manifest mlops.yaml -------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectSection(_Strict):
    name: str = Field(min_length=1)
    task: Literal["classification", "regression"]


class DataSection(_Strict):
    path: str = Field(min_length=1)
    target: str = Field(min_length=1)


class EvalSection(_Strict):
    primary_metric: str
    min_threshold: float


class Entrypoints(_Strict):
    train: str = "train.py"
    predict: str = "predict.py"


class Manifest(_Strict):
    project: ProjectSection
    data: DataSection
    evaluation: EvalSection
    entrypoints: Entrypoints = Entrypoints()
