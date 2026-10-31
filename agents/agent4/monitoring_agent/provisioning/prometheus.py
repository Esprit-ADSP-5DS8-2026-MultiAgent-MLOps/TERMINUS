"""
Génération des règles d'alerte Prometheus pour un projet.

Les règles dépendent du projet : la métrique principale (roc_auc, mae...), son sens
(plus haut = mieux, ou l'inverse), le seuil métier de mlops.yaml. On ne copie pas un fichier
générique : on le produit à partir de la spec, puis on le valide (promtool) avant de le charger.

Chaque alerte porte trois étiquettes utilisées partout ensuite (Alertmanager, agent, console) :
- severity  : info | warning | critical
- component : service | data | model     (qui est en cause)
- project   : le projet surveillé
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from ..runtime.target import MonitoringTarget
from ..schemas import HIGHER_IS_BETTER


def _rule(name: str, expr: str, severity: str, component: str, summary: str, description: str,
          for_: str = "0s", project: str = "") -> dict:
    return {
        "alert": name,
        "expr": expr,
        "for": for_,
        "labels": {"severity": severity, "component": component, "project": project},
        "annotations": {"summary": summary, "description": description},
    }


def build_rules(target: MonitoringTarget) -> dict:
    p = target.project
    m = target.mapping.primary_metric
    sel = f'project="{p}"'
    req = f'mlmon_http_requests_total{{{sel},route="/predict"}}'
    worse = "<" if m in HIGHER_IS_BETTER else ">"
    worse_txt = "sous" if m in HIGHER_IS_BETTER else "au-dessus de"

    def ratio(status_re: str) -> str:
        num = f'sum(rate(mlmon_http_requests_total{{{sel},route="/predict",status=~"{status_re}"}}[2m]))'
        return f"{num} / clamp_min(sum(rate({req}[2m])), 1e-9)"

    service = [
        _rule("MLGatewayDown", 'up{job="ml-gateway"} == 0', "critical", "service",
              "Gateway de monitoring injoignable", "Prometheus ne collecte plus /metrics depuis 30 s.", "30s", p),
        _rule("MLUpstreamDown", f"mlmon_upstream_up{{{sel}}} == 0", "critical", "service",
              "API du modèle indisponible", "La gateway n'obtient plus de réponse de /health du modèle.", "30s", p),
        _rule("MLHighErrorRate", f"{ratio('5..')} > 0.05", "critical", "service",
              "Erreurs serveur sur /predict", 'Taux d\'erreurs 5xx : {{ $value | humanizePercentage }}.', "1m", p),
        _rule("MLInvalidInputs", f"{ratio('4..')} > 0.05", "warning", "data",
              "Requêtes rejetées par le modèle (4xx)",
              'Le client envoie des données invalides : {{ $value | humanizePercentage }} des requêtes.', "1m", p),
        _rule("MLHighLatency",
              f'histogram_quantile(0.95, sum by (le) (rate(mlmon_http_request_duration_seconds_bucket{{{sel},route="/predict"}}[2m]))) > 0.5',
              "warning", "service", "Latence p95 élevée", "p95 = {{ $value | humanizeDuration }}.", "2m", p),
    ]
    data = [
        _rule("MLDatasetDrift", f"mlmon_dataset_drift_share{{{sel}}} >= 0.5", "warning", "data",
              "Drift sur la majorité des features",
              "{{ $value | humanizePercentage }} des features ont dérivé dans la dernière fenêtre.", project=p),
        _rule("MLFeatureDriftMajor", f"mlmon_feature_psi{{{sel}}} >= 0.25", "info", "data",
              "Drift majeur sur {{ $labels.feature }}", "PSI = {{ $value | printf \"%.3f\" }} (seuil 0.25).", project=p),
        _rule("MLPredictionDrift", f"mlmon_prediction_drifted{{{sel}}} == 1", "warning", "model",
              "La distribution des prédictions a changé",
              "Le modèle ne répond plus comme sur les données d'entraînement.", project=p),
        _rule("MLDataQualityWarning", f"mlmon_data_quality_severity{{{sel}}} == 2", "warning", "data",
              "Problème de qualité des données", "Voir le détail par feature dans le dashboard.", project=p),
        _rule("MLDataQualityCritical", f"mlmon_data_quality_severity{{{sel}}} >= 3", "critical", "data",
              "Problème critique de qualité des données",
              "Colonne manquante, type invalide ou forte hausse des valeurs manquantes / catégories inconnues.",
              project=p),
    ]
    model = [
        _rule("MLPerformanceDropEstimated",
              f'mlmon_performance_estimated{{{sel},metric="{m}"}} {worse} on(project, metric) mlmon_performance_bound{{{sel},metric="{m}"}}',
              "warning", "model", f"{m} estimé {worse_txt} la variabilité normale",
              f"{m} estimé sans labels = {{{{ $value | printf \"%.4f\" }}}}.", project=p),
        _rule("MLPerformanceDropRealized",
              f'mlmon_performance_realized{{{sel},metric="{m}"}} {worse} on(project, metric) mlmon_performance_bound{{{sel},metric="{m}"}}',
              "critical", "model", f"{m} mesuré {worse_txt} la variabilité normale",
              f"{m} mesuré avec les vrais labels = {{{{ $value | printf \"%.4f\" }}}}.", project=p),
    ]
    thr = target.mapping.min_threshold
    if thr is not None:
        model.append(_rule(
            "MLBelowBusinessThreshold", f'mlmon_performance_realized{{{sel},metric="{m}"}} {worse} {thr}',
            "critical", "model", f"{m} {worse_txt} le seuil métier ({thr})",
            f"Seuil fixé dans mlops.yaml : {thr}. Mesuré : {{{{ $value | printf \"%.4f\" }}}}.", project=p))

    return {"groups": [
        {"name": f"{p}-service", "interval": "10s", "rules": service},
        {"name": f"{p}-data", "interval": "10s", "rules": data},
        {"name": f"{p}-model", "interval": "10s", "rules": model},
    ]}


def write_rules(target: MonitoringTarget, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target.project}.rules.yml"
    header = f"# Généré par l'Agent 4 pour {target.project}. Ne pas éditer : relancer `provision`.\n"
    path.write_text(header + yaml.safe_dump(build_rules(target), sort_keys=False, allow_unicode=True, width=200),
                    encoding="utf-8")
    return path


def check_rules(path: Path) -> tuple[bool | None, str]:
    """Validation avec promtool (binaire local, sinon conteneur Docker). None = outil indisponible."""
    if shutil.which("promtool"):
        cmd = ["promtool", "check", "rules", str(path)]
    elif shutil.which("docker"):
        cmd = ["docker", "run", "--rm", "-v", f"{path.parent.resolve()}:/rules:ro", "--entrypoint", "promtool",
               "prom/prometheus:v3.5.0", "check", "rules", f"/rules/{path.name}"]
    else:
        return None, "promtool introuvable (ni binaire, ni Docker) : validation ignorée"
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"promtool non exécutable : {exc}"
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 and cmd[0] == "docker" and "SUCCESS" not in out and "FAILED" not in out:
        return None, f"Docker indisponible pour lancer promtool : {out[-300:]}"   # démon arrêté, image absente...
    return proc.returncode == 0, out
