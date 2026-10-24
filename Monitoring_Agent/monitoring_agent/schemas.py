"""
Schémas internes de l'Agent 4 (monitoring).

Les objets échangés avec l'orchestrateur restent ceux de common/schemas.py
(AgentInput, AgentResult, RetrainingRequest...). Ici on définit ce que le moteur
d'analyse produit : des rapports CALCULÉS, jamais rédigés par un LLM.
Le LLM (étape 3) lira ces rapports ; il ne les fabrique pas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MONITORING_SCHEMA_VERSION = "0.1"

Task = Literal["classification", "regression"]
Metric = Literal["accuracy", "precision", "recall", "f1", "roc_auc", "rmse", "mae", "r2"]
Severity = Literal["ok", "info", "warning", "critical"]
ColumnKind = Literal["numeric", "categorical"]

SEVERITY_RANK: dict[str, int] = {"ok": 0, "info": 1, "warning": 2, "critical": 3}
HIGHER_IS_BETTER: set[str] = {"accuracy", "precision", "recall", "f1", "roc_auc", "r2"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def max_severity(levels: list[str]) -> Severity:
    return max(levels, key=SEVERITY_RANK.__getitem__, default="ok")  # type: ignore[return-value]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Description des colonnes : qui est feature, cible, prédiction...
# ---------------------------------------------------------------------------


class ColumnMapping(Strict):
    """Rôle de chaque colonne. Dans le pipeline, il vient de project_spec.json."""

    task: Task
    features: list[str]
    target: str | None = None             # vérité terrain (arrive souvent en retard)
    prediction: str                       # sortie du modèle
    prediction_proba: str | None = None   # score de la classe positive (binaire)
    timestamp: str | None = None
    id: str | None = None
    primary_metric: Metric
    min_threshold: float | None = None    # seuil métier (mlops.yaml)


# ---------------------------------------------------------------------------
# Résultats d'une fenêtre
# ---------------------------------------------------------------------------


class FeatureDriftResult(Strict):
    feature: str
    kind: ColumnKind
    method: Literal["wasserstein", "jensen_shannon", "ks", "chi2"]
    score_type: Literal["distance", "p_value"]   # distance : grand = drift ; p-value : petit = drift
    score: float
    threshold: float
    drifted: bool
    psi: float                                   # toujours calculé : mesure commune et lisible
    psi_level: Literal["none", "moderate", "major"]
    reference_summary: dict[str, float] = {}
    current_summary: dict[str, float] = {}


class DatasetDrift(Strict):
    n_features: int
    n_drifted: int
    share_drifted: float
    drifted: bool
    drifted_features: list[str] = []


class QualityIssue(Strict):
    feature: str
    kind: Literal["missing_column", "missing_values", "type_mismatch", "unseen_category", "out_of_range"]
    value: float                       # taux observé dans la fenêtre
    reference_value: float | None = None
    severity: Severity
    detail: str = ""


class PerformanceResult(Strict):
    metric: Metric
    labels_available: bool
    realized: dict[str, float] | None = None     # calculé avec les vrais labels
    estimated: dict[str, float] | None = None    # estimé SANS labels (CBPE / DLE)
    estimation_method: Literal["cbpe", "dle"] | None = None
    reference_mean: float | None = None          # niveau habituel sur la référence
    reference_std: float | None = None
    bound: float | None = None                   # limite de la variabilité naturelle
    min_threshold: float | None = None           # seuil métier
    realized_degraded: bool | None = None
    estimated_degraded: bool | None = None


class Signal(Strict):
    """Un fait calculé qui mérite l'attention de l'agent."""

    kind: Literal[
        "dataset_drift", "feature_drift", "prediction_drift", "data_quality",
        "performance_drop_estimated", "performance_drop_realized", "concept_drift_suspected",
    ]
    severity: Severity
    message: str
    evidence: dict[str, float | str] = {}


class WindowInfo(Strict):
    index: int
    n_rows: int
    start: str | None = None
    end: str | None = None


class WindowReport(Strict):
    window: WindowInfo
    feature_drift: list[FeatureDriftResult]
    prediction_drift: FeatureDriftResult | None = None
    dataset_drift: DatasetDrift
    data_quality: list[QualityIssue] = []
    performance: PerformanceResult | None = None
    signals: list[Signal] = []
    severity: Severity = "ok"
    trigger_agent: bool = False        # True -> on réveille le LLM (étape 3)


class MonitoringReport(Strict):
    """monitoring_report.json : l'historique des fenêtres analysées."""

    schema_version: str = MONITORING_SCHEMA_VERSION
    project: str
    generated_at: datetime = Field(default_factory=utcnow)
    mapping: ColumnMapping
    reference_rows: int
    window_size: int
    windows: list[WindowReport]
    summary: dict[str, float | int | str | list[str] | None] = {}
