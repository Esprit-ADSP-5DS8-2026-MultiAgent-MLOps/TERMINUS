"""
Adaptateur de l'Agent 2 (MLflow) pour l'orchestrateur (contrat : common/schemas.py).

    REAL["agent2"] = "mlflow_agent.orchestrator_adapter:run"

Entrées du contrat : project_spec, data_profile, baseline (Agent 1), et retraining_request (Agent 4)
quand l'orchestrateur relance l'Agent 2 après un drift.
Sortie : mlflow_metadata.json. Statut NEEDS_APPROVAL quand le candidat passe le quality gate (et bat le
champion s'il y en a un) : l'orchestrateur demande alors à un humain de le promouvoir.

L'agent ne lance jamais le code du projet : il lit le CSV et entraîne ses propres pipelines scikit-learn
(voir executor.py). MLflow : MLFLOW_TRACKING_URI, sinon SQLite dans runs/mlflow/ à la racine du dépôt.

    mlflow ui --backend-store-uri sqlite:///runs/mlflow/mlflow.db     # pour voir les expériences
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pandas as pd

from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    MlflowMetadata,
    QualityGate,
    read_output,
    write_output,
)

from .config import ConfigError, resolve_llm
from .planner import default_plan, plan_experiment
from .schemas import (
    AnalysisReport,
    Constraints,
    DatasetAnalysis,
    ExistingImplementation,
    MissingValues,
    ProjectInfo,
    Status,
    TaskInfo,
    resolve_paths,
)

REPO = Path(__file__).resolve().parents[3]
MINIMIZE = {"rmse", "mae"}
CHECK_LABELS = {
    "meets_threshold": "score de test sous le seuil du manifeste",
    "beats_baseline": "ne bat pas le modèle naïf",
    "overfit_gap_ok": "écart entraînement / validation trop grand (sur-apprentissage)",
    "test_consistent_with_validation": "score de test trop bas par rapport à la validation",
}


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO / 'runs' / 'mlflow' / 'mlflow.db'}"


def _artifact_root() -> Path:
    return Path(os.environ.get("AGENT2_MLFLOW_ARTIFACTS") or REPO / "runs" / "mlflow" / "artifacts")


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri()
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def _ensure_experiment(client, name: str) -> None:
    """Expérience créée avec un dossier d'artefacts connu (sinon MLflow écrit ./mlruns là où on se trouve)."""
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=(_artifact_root() / name).resolve().as_uri())


def _champion_score(client, model_name: str, metric: str) -> float | None:
    """Score de test du champion actuel (alias « champion » du registre), s'il existe."""
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
        return client.get_run(version.run_id).data.metrics.get(f"test_{metric}")
    except Exception:  # noqa: BLE001 - pas de modèle ou pas d'alias : pas de champion
        return None


def promote(model_name: str, version: str) -> None:
    """Fait du modèle `version` le champion (à appeler une fois la mise en production faite)."""
    _client().set_registered_model_alias(model_name, "champion", version)


def _dataset(inp: AgentInput, spec, warnings: list[str]) -> Path:
    """Les données d'entraînement : celles du projet, ou les nouvelles données d'un réentraînement."""
    src = Path(inp.project_dir) / spec.data_path
    if "retraining_request" in inp.inputs:
        req = read_output(inp, "retraining_request")
        if req.new_data_path and Path(req.new_data_path).is_file():
            src = Path(req.new_data_path)
            warnings.append(f"réentraînement ({req.reason}) sur les nouvelles données : {src.name}")
        else:
            warnings.append(f"réentraînement ({req.reason}) demandé sans nouvelles données : données du projet")
    # L'agent n'accepte que des chemins sous un dossier de base : on lui donne une copie dans le workdir.
    dest = Path(inp.workdir) / "data" / "train.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".parquet", ".pq"}:   # nouvelles données de l'Agent 4
        pd.read_parquet(src).to_csv(dest, index=False)
    else:
        shutil.copyfile(src, dest)
    return dest


def _production_score(inp: AgentInput, metric: str, warnings: list[str]) -> float | None:
    """Réentraînement : le champion se mesure sur la période qui a déclenché la demande (labels de production),
    pas sur son ancien jeu de test : c'est sur ces données que le candidat doit faire mieux."""
    if "retraining_request" not in inp.inputs:
        return None
    perf = read_output(inp, "retraining_request").current_performance or {}
    if metric in perf:
        warnings.append(f"réentraînement : champion comparé à sa performance mesurée en production "
                        f"({metric} {perf[metric]:.4f})")
        return float(perf[metric])
    return None


def _report(spec, df: pd.DataFrame) -> AnalysisReport:
    """Le rapport v0.1 qu'attend l'agent, construit depuis le contrat et les données."""
    missing = df.isna().sum()
    balance = ({str(k): int(v) for k, v in df[spec.target].value_counts().items()}
               if spec.task == "classification" else None)
    return AnalysisReport(
        schema_version="0.1",
        project=ProjectInfo(project_path=".", dataset_path="data/train.csv", training_script=None),
        task=TaskInfo(type=spec.task, target_column=spec.target, objective=spec.project_name[:300],
                      primary_metric=spec.primary_metric, min_primary_metric=spec.min_threshold),
        dataset_analysis=DatasetAnalysis(
            rows=len(df), columns=df.shape[1],
            missing_values=MissingValues(total=int(missing.sum()),
                                         columns_affected=[str(c) for c in df.columns if missing[c] > 0]),
            class_distribution=balance,
            data_leakage_risks=[c.message for c in spec.carry_forward if "LEAK" in c.code.upper()]),
        existing_implementation=ExistingImplementation(
            training_code_available="train" in spec.entrypoints, preprocessing_available=True, existing_models=[]),
        constraints=Constraints(max_experiment_runs=int(os.environ.get("AGENT2_MAX_RUNS", "12")),
                                max_runtime_minutes=int(os.environ.get("AGENT2_MAX_MINUTES", "10"))),
        status=Status.READY,
        warnings=[c.message for c in spec.carry_forward][:20],
    )


def _plan(report: AnalysisReport, warnings: list[str]):
    if os.environ.get("AGENT2_NO_LLM", "").lower() in {"1", "true", "yes"}:
        warnings.append("plan par défaut (AGENT2_NO_LLM)")
        return default_plan(report), "fallback"
    try:
        from .llm_client import make_client

        llm, _ = resolve_llm(None)
        result = plan_experiment(report, make_client(llm))
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        warnings.append(f"LLM indisponible ({type(exc).__name__}: {str(exc)[:150]}) : plan par défaut")
        return default_plan(report), "fallback"
    warnings += [f"LLM : {e}"[:200] for e in result.errors]
    if result.source == "fallback":
        warnings.append("plan du LLM invalide ou absent : plan par défaut")
    return result.plan, result.source


def run(inp: AgentInput) -> AgentResult:
    started = time.monotonic()
    spec = read_output(inp, "project_spec")
    workdir = Path(inp.workdir)
    warnings: list[str] = []

    data = _dataset(inp, spec, warnings)
    report = _report(spec, pd.read_csv(data))
    (workdir / "analysis_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    paths = resolve_paths(report, workdir)
    plan, source = _plan(report, warnings)

    from .executor import (
        run_experiment,  # imports lourds (mlflow, optuna) seulement ici
    )

    client = _client()
    name = spec.project_name
    _ensure_experiment(client, name)
    champion = _champion_score(client, name, spec.primary_metric)
    production = _production_score(inp, spec.primary_metric, warnings)
    if production is not None and champion is not None:
        champion = production
    result = run_experiment(report, plan, paths, tracking_uri=tracking_uri(), experiment_name=name,
                            registered_model_name=name, warnings_in=warnings)
    outputs = {"experiment_result": str(workdir / "experiment_result.json"),
               "analysis_report": str(workdir / "analysis_report.json")}
    Path(outputs["experiment_result"]).write_text(result.model_dump_json(indent=2), encoding="utf-8")

    if result.status == "failed" or result.test_score is None:
        return AgentResult(agent="agent2", status=AgentStatus.FAILED, can_proceed=False, outputs=outputs,
                           reasons=result.errors or ["expérience MLflow en échec"], warnings=result.warnings,
                           duration_s=round(time.monotonic() - started, 3))

    metric, test, base = spec.primary_metric, result.test_score, result.baseline_test_score
    sign = -1 if metric in MINIMIZE else 1
    reasons = [CHECK_LABELS.get(k, k) for k, ok in (result.quality_gate.checks.items()
                                                       if result.quality_gate else []) if not ok]
    gate_ok = bool(result.quality_gate and result.quality_gate.passed) and not reasons
    champion_kept = gate_ok and champion is not None and sign * test <= sign * champion
    if champion_kept:
        reasons.append(f"ne bat pas le champion actuel ({metric} {test:.4f} contre {champion:.4f})")
    passed = gate_ok and not champion_kept

    if result.model_version and passed:  # une version refusée reste dans le registre, sans alias
        client.set_registered_model_alias(name, "candidate", result.model_version)
    diff = workdir / "instrumentation.md"
    diff.write_text("Aucune instrumentation du code du projet : l'Agent 2 ne l'exécute pas. Il entraîne ses "
                    "propres pipelines scikit-learn sur le CSV et suit tout dans MLflow "
                    f"(expérience « {name} », run {result.run_id}).\n", encoding="utf-8")
    meta = MlflowMetadata(
        tracking_uri=tracking_uri(), experiment_name=name, mlflow_run_id=result.run_id or "",
        model_name=name, model_version=result.model_version or "",
        model_uri=result.model_uri or f"runs:/{result.run_id}/model", alias="candidate",
        metrics={f"train_{metric}": result.train_score, f"validation_{metric}": result.validation_score,
                 f"test_{metric}": test, f"baseline_test_{metric}": base},
        params={"model": result.best_model_name or "", "plan_source": source,
                **{k: str(v) for k, v in result.best_params.items()}},
        instrumentation_diff_path=str(diff),
        quality_gate=QualityGate(primary_metric=metric, value=test, min_threshold=spec.min_threshold,
                                 baseline_value=base, champion_value=champion, passed=passed, reasons=reasons),
        champion_replacement_requested=passed,
    )
    outputs["mlflow_metadata"] = write_output(inp, "mlflow_metadata", meta)
    elapsed = round(time.monotonic() - started, 3)
    # Le modèle actuel du projet (score vu par le validateur) : comparaison indicative, les découpages
    # train/test diffèrent, donc on informe l'humain sans bloquer.
    current = read_output(inp, "baseline").observed_metrics.get(metric) if "baseline" in inp.inputs else None
    note = ""
    if current is not None and sign * test < sign * current:
        note = f" ; attention : moins bon que le modèle actuel du projet ({metric} {current:.4f})"
        result.warnings.append(note.removeprefix(" ; "))
    if champion_kept:
        # Pas un échec : le modèle en production reste le meilleur, rien à promouvoir ni à redéployer.
        return AgentResult(agent="agent2", status=AgentStatus.WARNING, can_proceed=False, outputs=outputs,
                           reasons=[f"le champion actuel reste en production : nouveau candidat {metric} "
                                    f"{test:.4f}, champion {champion:.4f}"],
                           warnings=result.warnings, duration_s=elapsed)
    if not passed:
        return AgentResult(agent="agent2", status=AgentStatus.FAILED, can_proceed=False, outputs=outputs,
                           reasons=[f"quality gate raté ({metric}={test:.4f}) : " + " ; ".join(reasons)],
                           warnings=result.warnings, duration_s=elapsed)
    return AgentResult(
        agent="agent2", status=AgentStatus.NEEDS_APPROVAL, can_proceed=False, outputs=outputs,
        reasons=[f"quality gate réussi ({result.best_model_name}, {metric}={test:.4f}, modèle naïf {base:.4f}"
                 + (f", champion {champion:.4f}" if champion is not None else "") + ") : promotion à approuver"
                 + note],
        warnings=result.warnings, duration_s=elapsed)
