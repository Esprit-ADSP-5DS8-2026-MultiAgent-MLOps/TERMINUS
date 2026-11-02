"""
Contrat commun de la plateforme Agentic MLOps.

Tous les objets échangés entre l'orchestrateur et les agents sont définis ici.
Règle d'équipe : on ne modifie ce fichier qu'avec l'accord du groupe,
et on incrémente SCHEMA_VERSION à chaque changement.

Principes :
- l'orchestrateur ne transporte que des CHEMINS et des statuts, jamais les données elles-mêmes ;
- chaque agent lit ses fichiers d'entrée et écrit ses fichiers de sortie dans son dossier de travail ;
- un agent ne lit que les fichiers de ce contrat (et le rapport du validateur), jamais le format
  interne d'un autre agent. S'il a son propre format, son adaptateur fait la traduction
  (agents/<agent>/.../orchestrator_adapter.py, voir agents/README.md).

Historique :
- 0.1 : premier contrat (orchestrateur + stubs).
- 0.2 : boucle de réparation (RepairResult), ProjectSpec complété pour l'Agent 3
  (model_artifact, interfaces, python_version), DeploymentResult.model_source,
  OUTPUTS / OUTPUT_SCHEMAS (qui produit quoi), read_output / write_output.

Source de vérité du rapport du validateur : validator/models.py.
Ici, on ne reprend que la partie `handoff` dont les agents ont besoin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.2"

Task = Literal["classification", "regression"]
Metric = Literal["accuracy", "precision", "recall", "f1", "roc_auc", "rmse", "mae", "r2"]
AgentName = Literal["validator", "repair", "agent1", "agent2", "agent3", "agent4"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Strict(BaseModel):
    """Base de nos schémas : un champ inconnu est une erreur."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 1. Entrée commune : ce que le validateur a déjà établi (section `handoff`)
# ---------------------------------------------------------------------------


class HandoffStatus(str, Enum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"


class CarryForward(BaseModel):
    """Avertissement transmis par le validateur aux agents suivants."""

    code: str  # ex. "HANDOFF.IMBALANCED_CLASSES"
    message: str
    source: str


class ValidatorHandoff(BaseModel):
    """Vue en lecture seule de `report["handoff"]`.

    extra="ignore" : si le validateur ajoute un champ, ce schéma ne casse pas.
    """

    model_config = ConfigDict(extra="ignore")

    status: HandoffStatus
    can_proceed: bool
    requires_human_confirmation: bool
    reason: str | None = None
    task: Task | None = None
    target: str | None = None
    primary_metric: Metric | None = None
    min_threshold: float | None = None
    data_path: str | None = None
    feature_columns: list[str] = []
    entrypoints: dict[str, str] = {}
    interfaces: dict[str, str] = {}
    model_artifact: str | None = None
    pinned_dependencies: dict[str, str] = {}
    observed_metrics: dict[str, float] = {}
    deterministic: bool | None = None
    carry_forward: list[CarryForward] = []

    # Projet rejeté = jamais exécuté : le validateur écrit null pour ces champs.
    @field_validator("feature_columns", "carry_forward", mode="before")
    @classmethod
    def _null_to_empty_list(cls, v):
        return [] if v is None else v

    @field_validator("entrypoints", "interfaces", "pinned_dependencies", "observed_metrics", mode="before")
    @classmethod
    def _null_to_empty_dict(cls, v):
        return {} if v is None else v


# ---------------------------------------------------------------------------
# 2. Interface commune des agents
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    OK = "OK"                          # terminé, on continue
    WARNING = "WARNING"                # terminé avec des points à surveiller
    NEEDS_APPROVAL = "NEEDS_APPROVAL"  # un humain doit décider avant de continuer
    FAILED = "FAILED"                  # échec : le pipeline s'arrête


class AgentInput(Strict):
    """Ce que l'orchestrateur donne à chaque agent."""

    run_id: str
    project_dir: str      # snapshot figé du projet validé (lecture seule)
    report_path: str      # compatibility_report.json
    workdir: str          # runs/<run_id>/<agent>/ : l'agent n'écrit QUE là
    inputs: dict[str, str] = {}  # sorties des agents précédents : nom -> chemin


class AgentResult(Strict):
    """Ce que chaque agent renvoie à l'orchestrateur."""

    schema_version: str = SCHEMA_VERSION
    agent: AgentName
    status: AgentStatus
    can_proceed: bool
    outputs: dict[str, str] = {}  # ex. {"project_spec": "runs/.../project_spec.json"}
    reasons: list[str] = []       # pourquoi FAILED / NEEDS_APPROVAL
    warnings: list[str] = []
    started_at: datetime = Field(default_factory=utcnow)
    duration_s: float | None = None


class Agent(Protocol):
    """Signature que chaque agent doit exposer : `def run(inp) -> AgentResult`."""

    def __call__(self, inp: AgentInput) -> AgentResult: ...


# ---------------------------------------------------------------------------
# 3. Agent 1 : analyse du projet
# ---------------------------------------------------------------------------


class ProjectSpec(Strict):
    """project_spec.json"""

    schema_version: str = SCHEMA_VERSION
    project_name: str
    project_hash: str               # repris du rapport : lie la spec au projet validé
    task: Task
    target: str
    feature_columns: list[str]
    primary_metric: Metric
    min_threshold: float
    data_path: str
    entrypoints: dict[str, str]
    pinned_dependencies: dict[str, str]
    # v0.2 : ce dont l'Agent 3 a besoin pour servir le modèle (repris du handoff du validateur)
    model_artifact: str | None = None            # ex. "model.joblib" (relatif au projet)
    interfaces: dict[str, str] = {}              # ex. {"predict": "predict(model, data) -> list"}
    python_version: str | None = None            # ex. "3.12"
    source: Literal["declared", "inferred"]  # mlops.yaml fourni, ou déduit par le LLM
    confidence: float = Field(1.0, ge=0, le=1)  # < 1 seulement si "inferred"
    carry_forward: list[CarryForward] = []


class ColumnProfile(Strict):
    name: str
    dtype: Literal["numeric", "categorical", "datetime", "text", "other"]
    missing_rate: float = Field(ge=0, le=1)
    n_unique: int
    # numériques
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    # catégorielles : fréquence de chaque modalité (référence pour le drift)
    frequencies: dict[str, float] | None = None


class DataProfile(Strict):
    """data_profile.json : aussi la référence de l'Agent 4 pour le drift."""

    schema_version: str = SCHEMA_VERSION
    rows: int
    target: str
    class_balance: dict[str, int] | None = None  # classification uniquement
    columns: list[ColumnProfile]


class Baseline(Strict):
    """baseline.json : le niveau à battre pour le quality gate."""

    schema_version: str = SCHEMA_VERSION
    primary_metric: Metric
    observed_metrics: dict[str, float]  # modèle du projet (vu par le validateur)
    naive_metrics: dict[str, float]     # modèle naïf (DummyClassifier/Regressor)


class ProposedManifest(Strict):
    """mlops.yaml proposé par l'Agent 1 quand le validateur dit AWAITING_CONFIRMATION."""

    task: Task
    target: str
    primary_metric: Metric
    min_threshold: float
    confidence: float = Field(ge=0, le=1)
    rationale: str


# ---------------------------------------------------------------------------
# 4. Agent 2 : MLflow et cycle de vie du modèle
# ---------------------------------------------------------------------------


class QualityGate(Strict):
    primary_metric: Metric
    value: float
    min_threshold: float
    baseline_value: float              # naive_metrics de baseline.json
    champion_value: float | None = None  # None s'il n'y a pas encore de champion
    passed: bool
    reasons: list[str] = []


class MlflowMetadata(Strict):
    """mlflow_metadata.json"""

    schema_version: str = SCHEMA_VERSION
    tracking_uri: str
    experiment_name: str
    mlflow_run_id: str
    model_name: str
    model_version: str
    model_uri: str                      # ex. "models:/credit-risk/3"
    alias: Literal["candidate", "champion"]
    metrics: dict[str, float]
    params: dict[str, str] = {}
    instrumentation_diff_path: str      # diff montré à l'humain
    quality_gate: QualityGate
    champion_replacement_requested: bool  # True -> gate humaine dans l'orchestrateur


# ---------------------------------------------------------------------------
# 5. Agent 3 : packaging et déploiement
# ---------------------------------------------------------------------------


class CheckResult(Strict):
    name: Literal["hadolint", "build", "trivy", "startup", "health", "inference"]
    passed: bool
    details: str = ""


class DeploymentResult(Strict):
    """deployment_result.json (deployment_config.yaml reste un fichier généré).

    L'Agent 3 déploie le modèle promu par l'Agent 2 (mlflow_metadata.model_uri).
    Repli : si MLflow n'a pas fourni de modèle, il sert l'artefact du projet (model_source).
    """

    schema_version: str = SCHEMA_VERSION
    model_uri: str                               # "models:/..." ou chemin de l'artefact en repli
    model_source: Literal["mlflow", "project_artifact"] = "mlflow"
    image: str                 # ex. "org/credit-risk"
    tag: str                   # tag versionné, jamais "latest" seul
    checks: list[CheckResult]
    pushed: bool               # False si un seul check a échoué
    endpoint: str | None = None  # ex. "http://localhost:8000"
    generated_files: dict[str, str] = {}  # "Dockerfile" -> chemin, etc.


# ---------------------------------------------------------------------------
# 6. Agent 4 : monitoring et réentraînement
# ---------------------------------------------------------------------------


class FeatureDrift(Strict):
    feature: str
    method: Literal["ks", "psi", "chi2", "evidently"]
    score: float
    drifted: bool


class RetrainingRequest(Strict):
    """retraining_request.json : une DEMANDE, jamais une décision."""

    schema_version: str = SCHEMA_VERSION
    model_uri: str
    reason: Literal["data_drift", "performance_drop", "manual"]
    drift: list[FeatureDrift] = []
    current_performance: dict[str, float] | None = None  # si les labels sont connus
    new_data_path: str | None = None
    diagnosis: str = ""
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# 7. Réparation (validateur v2) : boucle validateur -> réparation -> juge -> revalidation
# ---------------------------------------------------------------------------


class RepairChange(Strict):
    file: str               # chemin relatif au projet
    finding_code: str       # finding du validateur corrigé, ex. "DEPS.UNPINNED"
    description: str


class JudgeVerdict(Strict):
    approved: bool          # False : la réparation est jetée, le projet reste BLOCKED
    score: float | None = Field(None, ge=0, le=1)
    rationale: str


class RepairResult(Strict):
    """repair_result.json : le projet d'origine n'est JAMAIS modifié.

    L'agent de réparation écrit une copie corrigée dans son workdir (`repaired_project`),
    puis l'orchestrateur la fait revalider. La suite du pipeline travaille sur cette copie.
    """

    schema_version: str = SCHEMA_VERSION
    attempt: int = Field(ge=1)
    source_report: str          # rapport du validateur qui a motivé la réparation
    repaired_project: str       # dossier de la copie corrigée
    patch_path: str | None = None  # diff unifié, montré à l'humain
    changes: list[RepairChange]
    judge: JudgeVerdict


# ---------------------------------------------------------------------------
# 8. Décisions humaines (tracées par l'orchestrateur)
# ---------------------------------------------------------------------------


class HumanDecision(Strict):
    gate: Literal["manifest_confirmation", "champion_replacement", "retraining", "failure_review",
                  "repair_review", "validator_override"]
    decision: Literal["approve", "reject"]
    by: str
    comment: str = ""
    at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# 9. Qui produit quoi : noms des sorties et schéma de chacune
# ---------------------------------------------------------------------------

# Nom de sortie (clé de AgentResult.outputs) -> schéma du fichier JSON.
OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "project_spec": ProjectSpec,
    "data_profile": DataProfile,
    "baseline": Baseline,
    "proposed_manifest": ProposedManifest,
    "mlflow_metadata": MlflowMetadata,
    "deployment_result": DeploymentResult,
    "retraining_request": RetrainingRequest,
    "repair_result": RepairResult,
}

# Sorties obligatoires de chaque agent quand il réussit (statut OK, WARNING ou NEEDS_APPROVAL).
# D'autres sorties sont permises (fichiers de travail, rapports), elles ne sont pas vérifiées.
OUTPUTS: dict[str, list[str]] = {
    "agent1": ["project_spec", "data_profile", "baseline"],
    "propose_manifest": ["proposed_manifest"],
    "agent2": ["mlflow_metadata"],
    "agent3": ["deployment_result"],
    "agent4": [],                       # retraining_request seulement si un réentraînement est demandé
    "repair": ["repair_result", "repaired_project"],
}


def read_output(inp: AgentInput, name: str):
    """Lit une sortie d'un agent précédent et la valide avec son schéma.

        spec = read_output(inp, "project_spec")   # -> ProjectSpec
    """
    if name not in inp.inputs:
        raise KeyError(f"entrée « {name} » absente (reçues : {', '.join(sorted(inp.inputs)) or 'aucune'})")
    with open(inp.inputs[name], encoding="utf-8") as f:
        return OUTPUT_SCHEMAS[name].model_validate_json(f.read())


def write_output(inp: AgentInput, name: str, obj: BaseModel) -> str:
    """Écrit une sortie dans le workdir de l'agent (<workdir>/<name>.json) et renvoie son chemin.

        outputs["project_spec"] = write_output(inp, "project_spec", spec)
    """
    expected = OUTPUT_SCHEMAS.get(name)
    if expected is not None and not isinstance(obj, expected):
        raise TypeError(f"« {name} » doit être un {expected.__name__}, reçu {type(obj).__name__}")
    path = f"{inp.workdir.rstrip('/')}/{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj.model_dump_json(indent=2))
    return path
