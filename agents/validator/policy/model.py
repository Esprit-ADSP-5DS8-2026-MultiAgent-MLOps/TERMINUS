from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from agentic_core.schemas import Category


class NotApplicableCondition(BaseModel):
    id: str
    description: str


class ContractCheck(BaseModel):
    id: str
    level: Literal["structure", "code_env", "ml_contract", "security_exec"]
    title: str
    description: str
    requirement: Literal["mandatory", "recommended"] = "mandatory"
    on_fail: Literal["REPAIR_REQUIRED", "REJECTED", "NEEDS_REVIEW"] = "REPAIR_REQUIRED"
    severity: Literal["CRITICAL", "MAJOR", "MINOR"] = "MAJOR"
    nature: Literal["mechanical", "configuration", "logic", "methodological"] = "mechanical"
    category: Category = "other"
    not_applicable_when: list[NotApplicableCondition] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)   # contrôles dont l'échec peut CAUSER celui-ci (cascade, voir collapse_cascades)


class AdmissionContract(BaseModel):
    id: str
    version: str
    description: str = ""
    harness: Optional[str] = None                   # profil d'exécution (agentic_core.sandbox_exec.HARNESSES) ; None = pas d'exécution
    execution_required: bool = True
    sandbox_memory_mb: Optional[int] = None          # torch/tensorflow : la limite par défaut (4 Go d'espace d'adressage) est trop basse
    frameworks_in_scope: list[str] = Field(default_factory=list)
    frameworks_out_of_scope: list[str] = Field(default_factory=list)
    required_files: list[str] = Field(default_factory=list)  # STRUCT.REQUIRED_FILES : source de vérité structurée
    checks: list[ContractCheck]

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [c.id for c in self.checks]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"identifiants de contrôle dupliqués : {sorted(dup)}")
        if self.execution_required and not self.harness:
            raise ValueError("execution_required exige un `harness`")
        return self

    def check(self, check_id: str) -> Optional[ContractCheck]:
        return next((c for c in self.checks if c.id == check_id), None)

    @property
    def mandatory_ids(self) -> list[str]:
        return [c.id for c in self.checks if c.requirement == "mandatory"]


class ProtectedPattern(BaseModel):
    label: str
    regex: str


class RepairPolicy(BaseModel):
    auto_repair_natures: list[str] = ["mechanical", "configuration"]   # les autres natures exigent un humain
    protected_paths: list[str]
    scientific_patterns: list[ProtectedPattern]
    hardcoded_metric_regex: str
    metric_tolerance: float = 0.01
