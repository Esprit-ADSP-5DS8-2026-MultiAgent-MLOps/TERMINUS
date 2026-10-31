"""
Adaptateur de l'Agent 1 pour l'orchestrateur (contrat : common/schemas.py).

    REAL["agent1"] = "agents.agent1.orchestrator_adapter:run"
    REAL["propose_manifest"] = "agents.agent1.orchestrator_adapter:propose_manifest"

run(inp)               projet accepté par le validateur -> project_spec, data_profile, baseline
propose_manifest(inp)  projet sans mlops.yaml           -> proposed_manifest (validé par un humain)

L'agent travaille sur une copie du projet (load_project) : le projet validé n'est jamais modifié.
Ses améliorations de code sont des PROPOSITIONS (code_patch, improved_project) : la suite du pipeline
continue sur le projet validé, puisque le code modifié n'a pas été revalidé.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from common.envscope import agent_env
from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    Baseline,
    ColumnProfile,
    DataProfile,
    ProjectSpec,
    ProposedManifest,
    ValidatorHandoff,
    write_output,
)

from .runner import run_agent1

# Métrique et seuil proposés quand le projet n'a pas de mlops.yaml (l'humain confirme).
DEFAULT_METRIC = {"binary": ("roc_auc", 0.7), "multiclass": ("f1", 0.6), "regression": ("r2", 0.5)}


def _handoff(inp: AgentInput) -> tuple[dict, ValidatorHandoff]:
    report = json.loads(Path(inp.report_path).read_text(encoding="utf-8"))
    return report, ValidatorHandoff(**report["handoff"])


def _find_csv(project: Path, declared: str | None) -> Path | None:
    if declared and (project / declared).is_file():
        return project / declared
    candidates = [p for p in project.rglob("*.csv") if ".venv" not in p.parts and "__pycache__" not in p.parts]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def _python_version(project: Path) -> str | None:
    f = project / ".python-version"
    return (f.read_text(encoding="utf-8").strip() or None) if f.exists() else None


def _run_agent(inp: AgentInput, csv: Path) -> dict[str, Any]:
    """Lance l'Agent 1 ; ses artefacts (projet amélioré, patchs) vont dans le workdir."""
    artifacts = Path(inp.workdir) / "agent1"
    # Sa config charge agents/agent1/.env dans os.environ : on limite ces clés (GROQ...) au temps de l'appel.
    with agent_env(Path(__file__).resolve().parent / ".env"):
        out = run_agent1(inp.project_dir, str(csv), artifact_dir=str(artifacts))
    (Path(inp.workdir) / "agent1_output.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _extra_outputs(inp: AgentInput, out: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Rapport complet + propositions de l'agent, et ce qu'il faut en dire à l'humain."""
    outputs = {"agent1_report": str(Path(inp.workdir) / "agent1_output.json")}
    warnings: list[str] = []
    cq = out.get("code_quality", {})
    patch = Path(cq.get("patch_path") or "")
    if patch.is_file() and patch.stat().st_size:
        outputs["code_patch"] = str(patch)
        outputs["improved_project"] = cq.get("improved_project_path", "")
        warnings.append(f"qualité du code : {cq.get('issues_before', 0)} -> {cq.get('issues_after', 0)} problèmes ; "
                        f"{len(cq.get('changes', []))} modifications proposées (code_patch), non appliquées "
                        "au projet validé")
    ml_patch = Path((cq.get("ml_fixes") or {}).get("ml_patch_path") or "")
    if ml_patch.is_file() and ml_patch.stat().st_size:
        outputs["ml_patch"] = str(ml_patch)
        warnings.append("corrections ML proposées (ml_patch) : elles changent les résultats, à revoir par un humain")
    seen: set[str] = set()
    for f in (cq.get("ml_review") or {}).get("findings", []):
        text = str(f.get("message") or f.get("title") or f)
        if text not in seen and len(seen) < 3:
            seen.add(text)
            warnings.append(f"revue ML ({f.get('severity', '?')}) : {text}"[:300])
    for err in cq.get("tool_errors", [])[:2]:
        warnings.append(f"outil {err.get('node') or err.get('tool', '?')} : {err.get('message', err)}"[:300])
    if cq.get("failure_reason"):
        warnings.append(f"exécution de référence du projet : {cq['failure_reason']}"[:300])
    return {k: v for k, v in outputs.items() if v}, warnings


# ---------------------------------------------------------------------------
# Profil des données et modèle naïf
# ---------------------------------------------------------------------------


def _dtype(name: str, series: pd.Series, features: dict[str, list[str]]) -> str:
    if name in features.get("text", []):
        return "text"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series) or name in features.get("categorical", []):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _profile(df: pd.DataFrame, target: str, task: str, features: dict[str, list[str]]) -> DataProfile:
    columns = []
    for name in df.columns:
        s = df[name]
        kind = _dtype(str(name), s, features)
        numeric = kind == "numeric"
        columns.append(ColumnProfile(
            name=str(name),
            dtype=kind,
            missing_rate=float(s.isna().mean()),
            n_unique=int(s.nunique()),
            mean=float(s.mean()) if numeric else None,
            std=float(s.std()) if numeric else None,
            min=float(s.min()) if numeric else None,
            max=float(s.max()) if numeric else None,
            frequencies=None if numeric else {str(k): float(v) for k, v in s.value_counts(normalize=True).items()},
        ))
    balance = None
    if task == "classification":
        balance = {str(k): int(v) for k, v in df[target].value_counts().items()}
    return DataProfile(rows=len(df), target=target, class_balance=balance, columns=columns)


def _naive_metrics(y: pd.Series, task: str) -> dict[str, float]:
    """Score d'un modèle naïf (classe majoritaire / moyenne), calculé sur toutes les données."""
    y = y.dropna()
    if task == "regression":
        return {"r2": 0.0, "rmse": float(((y - y.mean()) ** 2).mean() ** 0.5),
                "mae": float((y - y.median()).abs().mean())}
    freq = y.value_counts(normalize=True)
    majority = freq.index[0]
    labels = sorted(freq.index, key=str)
    positive = labels[-1] if len(labels) == 2 else majority  # binaire : 1 / True / la 2e classe
    p = float(freq.get(positive, 0.0))
    hit = majority == positive
    return {"accuracy": float(freq.iloc[0]), "roc_auc": 0.5,
            "precision": p if hit else 0.0, "recall": 1.0 if hit else 0.0,
            "f1": (2 * p / (1 + p)) if hit else 0.0}


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------


def run(inp: AgentInput) -> AgentResult:
    started = time.monotonic()
    report, h = _handoff(inp)
    project = Path(inp.project_dir)
    csv = _find_csv(project, h.data_path)
    if csv is None or not (h.task and h.target and h.primary_metric and h.min_threshold is not None):
        return AgentResult(agent="agent1", status=AgentStatus.FAILED, can_proceed=False,
                           reasons=["données ou manifeste introuvables dans le handoff du validateur"],
                           duration_s=round(time.monotonic() - started, 3))

    out = _run_agent(inp, csv)
    outputs, warnings = _extra_outputs(inp, out)

    # Le manifeste déclaré (vérifié par le validateur) fait foi ; la détection de l'agent sert de contrôle.
    if out.get("target") and out["target"] != h.target:
        warnings.append(f"l'Agent 1 détecte la cible « {out['target']} », le manifeste déclare « {h.target} »")
    if out.get("problem_type") and out["problem_type"] != h.task:
        warnings.append(f"l'Agent 1 détecte « {out['problem_type']} », le manifeste déclare « {h.task} »")

    df = pd.read_csv(csv)
    spec = ProjectSpec(
        project_name=report["project"], project_hash=report["project_hash"], task=h.task, target=h.target,
        feature_columns=h.feature_columns or [c for c in df.columns if c != h.target],
        primary_metric=h.primary_metric, min_threshold=h.min_threshold,
        data_path=str(csv.relative_to(project)), entrypoints=h.entrypoints,
        pinned_dependencies=h.pinned_dependencies, model_artifact=h.model_artifact, interfaces=h.interfaces,
        python_version=_python_version(project), source="declared", carry_forward=h.carry_forward,
    )
    features = out.get("features") or {}
    outputs["project_spec"] = write_output(inp, "project_spec", spec)
    outputs["data_profile"] = write_output(inp, "data_profile", _profile(df, h.target, h.task, features))
    outputs["baseline"] = write_output(inp, "baseline", Baseline(
        primary_metric=h.primary_metric, observed_metrics=h.observed_metrics,
        naive_metrics=_naive_metrics(df[h.target], h.task)))

    return AgentResult(agent="agent1", status=AgentStatus.WARNING if warnings else AgentStatus.OK,
                       can_proceed=True, outputs=outputs, warnings=warnings,
                       duration_s=round(time.monotonic() - started, 3))


def propose_manifest(inp: AgentInput) -> AgentResult:
    started = time.monotonic()
    _, h = _handoff(inp)
    csv = _find_csv(Path(inp.project_dir), h.data_path)
    if csv is None:
        return AgentResult(agent="agent1", status=AgentStatus.FAILED, can_proceed=False,
                           reasons=["aucun fichier CSV dans le projet : impossible de proposer un manifeste"],
                           duration_s=round(time.monotonic() - started, 3))
    out = _run_agent(inp, csv)
    outputs, warnings = _extra_outputs(inp, out)
    if not out.get("target") or out.get("problem_type") not in ("classification", "regression"):
        return AgentResult(agent="agent1", status=AgentStatus.FAILED, can_proceed=False, outputs=outputs,
                           reasons=["l'Agent 1 n'a pas pu déterminer la cible et la tâche"], warnings=warnings,
                           duration_s=round(time.monotonic() - started, 3))

    kind = out["subtype"] if out["problem_type"] == "classification" else "regression"
    metric, threshold = DEFAULT_METRIC.get(kind or "binary", ("roc_auc", 0.7))
    proposal = ProposedManifest(
        task=out["problem_type"], target=out["target"], primary_metric=metric, min_threshold=threshold,
        confidence=float(out.get("confidence", 0.0)),
        rationale=(f"cible « {out['target']} » détectée par l'Agent 1 dans le code et les données "
                   f"(confiance {float(out.get('confidence', 0.0)):.2f}), "
                   f"tâche {out['problem_type']}{' ' + out['subtype'] if out.get('subtype') else ''} ; "
                   f"métrique et seuil par défaut ({metric} >= {threshold}), à ajuster ; données : "
                   f"{csv.relative_to(inp.project_dir)}"),
    )
    outputs["proposed_manifest"] = write_output(inp, "proposed_manifest", proposal)
    return AgentResult(agent="agent1", status=AgentStatus.NEEDS_APPROVAL, can_proceed=False, outputs=outputs,
                       reasons=["mlops.yaml absent : manifeste proposé par l'Agent 1, confirmation humaine requise"],
                       warnings=warnings, duration_s=round(time.monotonic() - started, 3))
