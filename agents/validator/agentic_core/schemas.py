"""Schémas communs v2 (à figer en premier : ils débloquent toute l'équipe, orchestrateur compris).

Trois rapports, tous scellés (sha256) et chaînés par lignage :
  ValidatorReport -> RepairReport -> JudgeVerdict -> (revalidation) ValidatorReport -> ...

Règles de conception :
  * le Validator ne « déclare » pas son statut : il le reçoit du moteur de politique (package `policy`) ;
  * le Judge ne produit JAMAIS de décision d'admission (aucun champ `status` d'admission dans son verdict) ;
  * les erreurs d'infrastructure (quota, timeout, budget) sont des STATUTS (`run_status`, `failure`), pas des exceptions ;
  * chaque rapport porte un `next_action` pour le routage par l'orchestrateur.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from validator.models import Handoff  # le handoff existant est conservé tel quel

SCHEMA_VERSION = "2.0"


# ----------------------------------------------------------------------------- énumérations
class AdmissionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_CHECKED = "NOT_CHECKED"       # la grande nouveauté : ce qui n'a PAS été vérifié est dit explicitement


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class Nature(str, Enum):
    MECHANICAL = "mechanical"
    CONFIGURATION = "configuration"
    LOGIC = "logic"
    METHODOLOGICAL = "methodological"


class RunStatus(str, Enum):
    OK = "OK"
    FAILED_CLOSED = "FAILED_CLOSED"   # l'agent n'a pas pu conclure : jamais de décision favorable par défaut


class FailureKind(str, Enum):
    LLM_ERROR = "LLM_ERROR"
    QUOTA = "QUOTA"
    TIMEOUT = "TIMEOUT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    MAX_STEPS = "MAX_STEPS"
    NO_TOOL_USE = "NO_TOOL_USE"
    INVALID_SUBMISSION = "INVALID_SUBMISSION"
    INPUT_INVALID = "INPUT_INVALID"
    INTERNAL = "INTERNAL"


ADMISSION_RANK = {AdmissionStatus.ACCEPTED: 0, AdmissionStatus.NEEDS_REVIEW: 1,
                  AdmissionStatus.REPAIR_REQUIRED: 2, AdmissionStatus.REJECTED: 3}

Category = Literal["structure", "dependencies", "code", "data", "ml_contract", "security", "execution", "reproducibility", "other"]


class Failure(BaseModel):
    kind: FailureKind
    message: str


class LineageRef(BaseModel):
    report_type: Literal["validator", "repair", "judge"]
    report_id: str
    report_sha256: Optional[str] = None
    project_hash: Optional[str] = None


# ----------------------------------------------------------------------------- Validator
class Check(BaseModel):
    check_id: str                                   # id du catalogue du contrat, ex. ENV.DEPENDENCIES_PINNED
    name: str = ""
    level: str = ""                                 # structure | code_env | ml_contract | security_exec
    status: CheckStatus
    severity: Optional[Severity] = None
    file: Optional[str] = None
    line: Optional[int] = None
    evidence: Optional[str] = None
    na_reason: Optional[str] = None                 # id de condition « non applicable » prévue par le contrat
    note: Optional[str] = None
    source: Optional[Literal["python", "llm", "harness", "policy"]] = None
    origin: Optional[Literal["candidate", "harness", "sandbox", "validator"]] = None
    caused_by: Optional[list[str]] = None           # ids des contrôles racines dont ce contrôle n'est que la conséquence (cascade)
    # --- renseignés par le moteur de politique ---
    mandatory: Optional[bool] = None
    effective_status: Optional[CheckStatus] = None  # ex. un contrôle recommandé en FAIL devient WARNING
    grounded: Optional[bool] = None                 # la preuve cite-t-elle bien une sortie d'outil ?


class Issue(BaseModel):
    issue_id: str
    check_id: str
    category: Category = "other"
    severity: Severity = Severity.MAJOR
    file: Optional[str] = None
    line: Optional[int] = None
    description: str
    suggested_fix: Optional[str] = None
    nature: Nature = Nature.MECHANICAL
    evidence: Optional[str] = None


class Framework(BaseModel):
    name: str
    evidence: Optional[str] = None


class ProjectSpec(BaseModel):
    """Ce que le Validator a compris du projet (tâche, cible, métrique) : sert au handoff et à la comparaison avant/après."""
    task: Optional[Literal["classification", "regression"]] = None
    target: Optional[str] = None
    primary_metric: Optional[str] = None
    min_threshold: Optional[float] = None
    data_path: Optional[str] = None
    deduced: dict[str, str] = Field(default_factory=dict)   # slot -> preuve, pour tout ce que le manifeste ne donnait pas


class PolicyOutcome(BaseModel):
    contract_id: str
    contract_version: str
    status: AdmissionStatus
    can_proceed: bool
    reasons: list[str] = Field(default_factory=list)
    blocking_checks: list[str] = Field(default_factory=list)
    review_checks: list[str] = Field(default_factory=list)
    warning_checks: list[str] = Field(default_factory=list)


class _Sealed(BaseModel):
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""
    run_status: RunStatus = RunStatus.OK
    failure: Optional[Failure] = None
    lineage: list[LineageRef] = Field(default_factory=list)     # chaîne complète, du plus ancien au parent direct
    llm: dict[str, Any] = Field(default_factory=dict)
    report_sha256: Optional[str] = None


class ValidatorReport(_Sealed):
    report_type: Literal["validator"] = "validator"
    report_id: str = ""
    project: str
    project_hash: Optional[str] = None
    hashed_files: int = 0
    snapshot_path: Optional[str] = None
    policy_version: str = ""
    contract_id: str = ""
    project_type: str = "unknown"                   # « inconnu » plutôt qu'inventé
    frameworks: list[Framework] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)  # ce que l'analyse n'a pas pu couvrir
    project_spec: Optional[ProjectSpec] = None
    status: AdmissionStatus
    can_proceed: bool = False
    policy: Optional[PolicyOutcome] = None
    next_action: Literal["PROCEED", "REPAIR", "HUMAN_REVIEW", "STOP"] = "HUMAN_REVIEW"
    handoff: Optional[Handoff] = None
    observed_metrics: Optional[dict[str, Any]] = None
    sandbox: Optional[dict[str, Any]] = None
    environment: Optional[dict[str, Any]] = None      # environnement d'exécution vérifié (venv isolé, dépendances conformes)
    steps: list[dict[str, Any]] = Field(default_factory=list)   # étapes effectuées, dans l'ordre (phases + appels d'outils)
    summary: Optional[str] = None
    duration_s: float = 0.0


# ----------------------------------------------------------------------------- Repair
class Modification(BaseModel):
    patch_id: str
    file: str
    issue_id: Optional[str] = None
    action: Literal["replace", "create", "rewrite", "delete", "revert"]
    reason: str = ""
    diff_ref: Optional[str] = None                  # ex. patches/p0001.diff (relatif au dossier de métadonnées)
    risk: Literal["low", "medium", "high", "scientific"] = "low"


class TestResult(BaseModel):
    __test__ = False                                # pas une classe de test pour pytest
    name: str
    status: Literal["PASS", "FAIL", "SKIPPED", "ERROR"]
    command: str = ""
    exit_code: Optional[int] = None
    output_excerpt: str = ""


class UnresolvedIssue(BaseModel):
    issue_id: str
    reason: str
    needs_human: bool = False


class ScopeFlag(BaseModel):
    file: str
    kind: Literal["protected_path", "scientific_change", "hardcoded_metric", "secret_added", "source_modified"]
    detail: str
    blocked: bool = False                           # True : l'écriture a été refusée ; False : appliquée mais à approuver


class MetricsComparison(BaseModel):
    status: Literal["COMPARED", "AFTER_ONLY", "NOT_AVAILABLE"]
    reason: Optional[str] = None
    before: Optional[dict[str, float]] = None
    after: Optional[dict[str, float]] = None
    delta: Optional[dict[str, float]] = None
    tolerance: float = 0.0
    behavior_changed: Optional[bool] = None


class PatchRecord(BaseModel):
    patch_id: str
    files: list[str]
    issue_id: Optional[str] = None
    event: Literal["applied", "rolled_back"] = "applied"
    diff: str = ""


class RepairReport(_Sealed):
    report_type: Literal["repair"] = "repair"
    repair_id: str = ""
    project: str
    status: Literal["REPAIRED", "PARTIAL", "NOTHING_TO_DO", "ESCALATED", "FAILED"]
    attempt: int = 1
    source_project_hash: Optional[str] = None
    repaired_project_hash: Optional[str] = None
    workspace_path: Optional[str] = None
    meta_path: Optional[str] = None
    source_validator_status: Optional[str] = None
    modifications: list[Modification] = Field(default_factory=list)
    patches: list[PatchRecord] = Field(default_factory=list)      # historique (appliqués, annulés)
    tests: list[TestResult] = Field(default_factory=list)
    unresolved_issues: list[UnresolvedIssue] = Field(default_factory=list)
    scope_flags: list[ScopeFlag] = Field(default_factory=list)
    requires_human_approval: bool = False
    metrics_comparison: Optional[MetricsComparison] = None
    requires_judge: bool = False
    requires_revalidation: bool = False
    next_action: Literal["JUDGE", "ESCALATE", "NONE"] = "ESCALATE"
    removed_artifacts: list[str] = Field(default_factory=list)
    diff: str = ""
    summary: str = ""


# ----------------------------------------------------------------------------- Judge
CriterionName = Literal["correctness", "completeness", "regression_risk", "evidence_quality", "scope_compliance"]
CRITERIA: tuple[str, ...] = ("correctness", "completeness", "regression_risk", "evidence_quality", "scope_compliance")


class Criterion(BaseModel):
    name: CriterionName
    result: Literal["PASS", "CONCERN", "FAIL"]
    rationale: str
    evidence: Optional[str] = None                  # extrait EXACT d'un diff ou d'une sortie de test (vérifié)
    grounded: Optional[bool] = None


class JudgeVerdict(_Sealed):
    """Verdict sur UNE réparation. Aucune décision d'admission : c'est le rôle exclusif du Validator + moteur de politique."""
    report_type: Literal["judge"] = "judge"
    verdict_id: str = ""
    repair_id: str
    project: str = ""
    verdict: Literal["ACCEPT", "REVISE", "REJECT"]
    criteria: list[Criterion] = Field(default_factory=list)
    evidence_examined: list[str] = Field(default_factory=list)
    remaining_concerns: list[str] = Field(default_factory=list)
    reason: str = ""
    next_action: Literal["REVALIDATE", "RETRY", "ESCALATE", "ROLLBACK"]
    judge_model: Optional[str] = None
    repair_model: Optional[str] = None
    independent_model: Optional[bool] = None        # le Judge n'est-il PAS le modèle du Repair ?
    overridden_by_invariant: Optional[str] = None


Report = ValidatorReport | RepairReport | JudgeVerdict


# ----------------------------------------------------------------------------- scellé, lignage, export
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def canonical(report: BaseModel) -> str:
    return json.dumps(report.model_dump(mode="json", exclude={"report_sha256"}), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))


def seal(report):
    """Sceau d'intégrité : détecte toute modification (ce n'est PAS une signature : aucun secret)."""
    report.report_sha256 = hashlib.sha256(canonical(report).encode()).hexdigest()
    return report


def verify_seal(report) -> bool:
    return bool(report.report_sha256) and report.report_sha256 == hashlib.sha256(canonical(report).encode()).hexdigest()


def report_id_of(report) -> str:
    return {"validator": "report_id", "repair": "repair_id", "judge": "verdict_id"}[report.report_type] and \
        getattr(report, {"validator": "report_id", "repair": "repair_id", "judge": "verdict_id"}[report.report_type])


def lineage_ref(report) -> LineageRef:
    return LineageRef(report_type=report.report_type, report_id=report_id_of(report), report_sha256=report.report_sha256,
                      project_hash=getattr(report, "project_hash", None) or getattr(report, "source_project_hash", None))


def chain_from(parent) -> list[LineageRef]:
    """Lignage d'un nouveau rapport dont `parent` est le prédécesseur direct : toute la chaîne se retrouve."""
    return [] if parent is None else [*parent.lineage, lineage_ref(parent)]


MODELS: dict[str, type[BaseModel]] = {"validator_report": ValidatorReport, "repair_report": RepairReport, "judge_verdict": JudgeVerdict}


def load_report(source):
    """Charge un rapport (objet, dict ou chemin JSON) selon son `report_type`."""
    if isinstance(source, BaseModel):
        return source
    if not isinstance(source, dict):
        from pathlib import Path
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    kind = {"validator": ValidatorReport, "repair": RepairReport, "judge": JudgeVerdict}.get(source.get("report_type"))
    if kind is None:
        raise ValueError(f"report_type inconnu : {source.get('report_type')!r}")
    return kind.model_validate(source)


def schema_json(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False, sort_keys=True)
