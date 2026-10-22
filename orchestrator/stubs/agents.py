"""
Faux agents (stubs) en attendant les vrais.

Chaque stub respecte exactement le contrat : il reçoit un AgentInput,
écrit des fichiers valides (schémas de common/schemas.py) dans son workdir,
et renvoie un AgentResult. Quand un binôme livre son agent, on remplace
la fonction correspondante dans orchestrator/graph.py (dictionnaire AGENTS).

Pour simuler un drift (et tester la boucle de réentraînement) :
    ORCH_SIMULATE_DRIFT=1 python -m orchestrator run ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    Baseline,
    CheckResult,
    ColumnProfile,
    DataProfile,
    DeploymentResult,
    FeatureDrift,
    MlflowMetadata,
    ProjectSpec,
    ProposedManifest,
    QualityGate,
    RetrainingRequest,
    ValidatorHandoff,
)

STUB_WARNING = "stub : résultat fictif, à remplacer par le vrai agent"


def _write(path: Path, model: BaseModel) -> str:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def _load_report(inp: AgentInput) -> tuple[dict, ValidatorHandoff]:
    report = json.loads(Path(inp.report_path).read_text(encoding="utf-8"))
    return report, ValidatorHandoff(**report["handoff"])


# ---------------------------------------------------------------------------
# Agent 1 : analyse (cas normal)
# ---------------------------------------------------------------------------


def agent1_stub(inp: AgentInput) -> AgentResult:
    report, h = _load_report(inp)
    wd = Path(inp.workdir)

    spec = ProjectSpec(
        project_name=report["project"],
        project_hash=report["project_hash"],
        task=h.task,
        target=h.target,
        feature_columns=h.feature_columns,
        primary_metric=h.primary_metric,
        min_threshold=h.min_threshold,
        data_path=h.data_path,
        entrypoints=h.entrypoints,
        pinned_dependencies=h.pinned_dependencies,
        source="declared",
        carry_forward=h.carry_forward,
    )

    df = pd.read_csv(Path(inp.project_dir) / h.data_path)
    columns = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        columns.append(
            ColumnProfile(
                name=col,
                dtype="numeric" if numeric else "categorical",
                missing_rate=float(s.isna().mean()),
                n_unique=int(s.nunique()),
                mean=float(s.mean()) if numeric else None,
                std=float(s.std()) if numeric else None,
                min=float(s.min()) if numeric else None,
                max=float(s.max()) if numeric else None,
                frequencies=None if numeric else {
                    str(k): float(v) for k, v in s.value_counts(normalize=True).items()
                },
            )
        )
    balance = None
    if h.task == "classification":
        balance = {str(k): int(v) for k, v in df[h.target].value_counts().items()}
    profile = DataProfile(rows=len(df), target=h.target, class_balance=balance, columns=columns)

    naive = {"roc_auc": 0.5} if h.primary_metric == "roc_auc" else {h.primary_metric: 0.0}
    baseline = Baseline(
        primary_metric=h.primary_metric,
        observed_metrics=h.observed_metrics,
        naive_metrics=naive,
    )

    return AgentResult(
        agent="agent1",
        status=AgentStatus.OK,
        can_proceed=True,
        outputs={
            "project_spec": _write(wd / "project_spec.json", spec),
            "data_profile": _write(wd / "data_profile.json", profile),
            "baseline": _write(wd / "baseline.json", baseline),
        },
        warnings=[STUB_WARNING],
    )


# ---------------------------------------------------------------------------
# Agent 1 : proposition de mlops.yaml (quand le validateur dit AWAITING_CONFIRMATION)
# ---------------------------------------------------------------------------


def propose_manifest_stub(inp: AgentInput) -> AgentResult:
    _, h = _load_report(inp)
    proposal = ProposedManifest(
        task=h.task or "classification",
        target=h.target or "target",
        primary_metric=h.primary_metric or "roc_auc",
        min_threshold=h.min_threshold or 0.7,
        confidence=0.5,
        rationale="stub : valeurs par défaut, le vrai agent les déduira du code et des données",
    )
    return AgentResult(
        agent="agent1",
        status=AgentStatus.NEEDS_APPROVAL,
        can_proceed=False,
        outputs={"proposed_manifest": _write(Path(inp.workdir) / "proposed_manifest.json", proposal)},
        reasons=["mlops.yaml absent : manifeste proposé, confirmation humaine requise"],
        warnings=[STUB_WARNING],
    )


# ---------------------------------------------------------------------------
# Agent 2 : MLflow et quality gate
# ---------------------------------------------------------------------------


def agent2_stub(inp: AgentInput) -> AgentResult:
    spec = ProjectSpec.model_validate_json(Path(inp.inputs["project_spec"]).read_text())
    base = Baseline.model_validate_json(Path(inp.inputs["baseline"]).read_text())
    wd = Path(inp.workdir)

    metric = spec.primary_metric
    value = base.observed_metrics.get(metric, 0.0)
    baseline_value = base.naive_metrics.get(metric, 0.0)
    reasons = []
    if value < spec.min_threshold:
        reasons.append(f"{metric}={value:.4f} < seuil {spec.min_threshold}")
    if value <= baseline_value:
        reasons.append(f"{metric}={value:.4f} ne bat pas le modèle naïf ({baseline_value})")
    gate = QualityGate(
        primary_metric=metric,
        value=value,
        min_threshold=spec.min_threshold,
        baseline_value=baseline_value,
        passed=not reasons,
        reasons=reasons,
    )

    diff = wd / "instrumentation.diff"
    diff.write_text("# stub : aucun diff, le vrai agent ajoutera mlflow.autolog()\n")
    meta = MlflowMetadata(
        tracking_uri="stub://mlflow",
        experiment_name=spec.project_name,
        mlflow_run_id=f"stub-{inp.run_id}",
        model_name=spec.project_name,
        model_version="1",
        model_uri=f"models:/{spec.project_name}/1",
        alias="candidate",
        metrics=base.observed_metrics,
        instrumentation_diff_path=str(diff),
        quality_gate=gate,
        champion_replacement_requested=gate.passed,
    )
    outputs = {"mlflow_metadata": _write(wd / "mlflow_metadata.json", meta)}

    if not gate.passed:
        return AgentResult(agent="agent2", status=AgentStatus.FAILED, can_proceed=False,
                           outputs=outputs, reasons=reasons, warnings=[STUB_WARNING])
    return AgentResult(
        agent="agent2",
        status=AgentStatus.NEEDS_APPROVAL,
        can_proceed=False,
        outputs=outputs,
        reasons=[f"quality gate réussi ({metric}={value:.4f}) : promotion en champion à approuver"],
        warnings=[STUB_WARNING],
    )


# ---------------------------------------------------------------------------
# Agent 3 : packaging et déploiement
# ---------------------------------------------------------------------------


def agent3_stub(inp: AgentInput) -> AgentResult:
    meta = MlflowMetadata.model_validate_json(Path(inp.inputs["mlflow_metadata"]).read_text())
    checks = [CheckResult(name=n, passed=True, details="stub")
              for n in ("hadolint", "build", "trivy", "startup", "health", "inference")]
    dep = DeploymentResult(
        model_uri=meta.model_uri,
        image=f"stub/{meta.model_name}",
        tag=f"v{meta.model_version}-{inp.run_id}",
        checks=checks,
        pushed=all(c.passed for c in checks),
        endpoint=None,
    )
    return AgentResult(
        agent="agent3",
        status=AgentStatus.OK,
        can_proceed=True,
        outputs={"deployment_result": _write(Path(inp.workdir) / "deployment_result.json", dep)},
        warnings=[STUB_WARNING],
    )


# ---------------------------------------------------------------------------
# Agent 4 : monitoring
# ---------------------------------------------------------------------------


def agent4_stub(inp: AgentInput) -> AgentResult:
    dep = DeploymentResult.model_validate_json(Path(inp.inputs["deployment_result"]).read_text())
    if os.environ.get("ORCH_SIMULATE_DRIFT") != "1":
        return AgentResult(agent="agent4", status=AgentStatus.OK, can_proceed=True,
                           warnings=[STUB_WARNING])

    req = RetrainingRequest(
        model_uri=dep.model_uri,
        reason="data_drift",
        drift=[FeatureDrift(feature="amount", method="psi", score=0.31, drifted=True)],
        diagnosis="stub : drift simulé sur 'amount'",
    )
    return AgentResult(
        agent="agent4",
        status=AgentStatus.WARNING,
        can_proceed=True,
        outputs={"retraining_request": _write(Path(inp.workdir) / "retraining_request.json", req)},
        warnings=[STUB_WARNING, "drift détecté : demande de réentraînement"],
    )
