"""
Modèle d'adaptateur : à copier dans le paquet de son agent sous le nom orchestrator_adapter.py.

L'adaptateur est la seule partie de l'agent qui connaît l'orchestrateur. Il :
1. lit les fichiers du contrat (common/schemas.py) donnés par l'orchestrateur ;
2. les traduit dans le format interne de l'agent et lance l'agent ;
3. traduit le résultat en fichiers du contrat, écrits dans inp.workdir ;
4. renvoie un AgentResult. Une exception est permise (l'orchestrateur la transforme en FAILED),
   mais un échec expliqué dans reasons est plus utile à l'humain et au superviseur.

Vérifier avant de l'ajouter dans orchestrator/registry.py :
    python -m orchestrator check-agent agent2 mlflow_agent.orchestrator_adapter:run

Exemple ci-dessous : l'Agent 2 (adapter les noms et les champs à son agent).
"""

from __future__ import annotations

import time
from pathlib import Path

from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    MlflowMetadata,
    QualityGate,
    read_output,
    write_output,
)


def run(inp: AgentInput) -> AgentResult:
    started = time.monotonic()

    # 1. Entrées du contrat (jamais le format interne d'un autre agent)
    spec = read_output(inp, "project_spec")       # ProjectSpec
    baseline = read_output(inp, "baseline")       # Baseline
    project = Path(inp.project_dir)               # copie figée du projet : lecture seule
    workdir = Path(inp.workdir)                   # le seul dossier où l'agent écrit

    # 2. Traduction vers le format interne puis appel de l'agent (à remplacer)
    #    report = build_internal_report(spec, baseline, project)
    #    internal = run_experiment(report, output_dir=workdir)
    del project, workdir

    # 3. Résultat -> fichiers du contrat (valeurs à reprendre de `internal`)
    gate = QualityGate(primary_metric=spec.primary_metric, value=0.0, min_threshold=spec.min_threshold,
                       baseline_value=baseline.naive_metrics.get(spec.primary_metric, 0.0), passed=False,
                       reasons=["modèle d'adaptateur : remplacer par le vrai résultat"])
    meta = MlflowMetadata(tracking_uri="", experiment_name=spec.project_name, mlflow_run_id="",
                          model_name=spec.project_name, model_version="", model_uri="", alias="candidate",
                          metrics={}, instrumentation_diff_path="", quality_gate=gate,
                          champion_replacement_requested=False)
    outputs = {"mlflow_metadata": write_output(inp, "mlflow_metadata", meta)}

    # 4. Statut : OK / WARNING pour continuer, NEEDS_APPROVAL pour une gate humaine, FAILED pour arrêter
    if not gate.passed:
        return AgentResult(agent="agent2", status=AgentStatus.FAILED, can_proceed=False, outputs=outputs,
                           reasons=gate.reasons, duration_s=round(time.monotonic() - started, 3))
    return AgentResult(agent="agent2", status=AgentStatus.NEEDS_APPROVAL, can_proceed=False, outputs=outputs,
                       reasons=["quality gate réussi : promotion en champion à approuver"],
                       duration_s=round(time.monotonic() - started, 3))
