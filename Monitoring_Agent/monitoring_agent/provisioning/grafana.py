"""
Génération du dashboard Grafana d'UN projet (JSON model, chargé au démarrage par provisioning).

Le dashboard est construit à partir de la spec : nom du projet, métrique principale, son sens,
la limite de variabilité calculée sur la référence (colore les seuils), le seuil métier.
Quatre bandeaux : vue d'ensemble · modèle · données · service.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..analysis import Analyzer
from ..config import AnalyzerConfig
from ..runtime.target import MonitoringTarget
from ..schemas import HIGHER_IS_BETTER

DS = {"type": "prometheus", "uid": "prometheus"}
RED, ORANGE, GREEN, BLUE, GREY = "#E5484D", "#F5A524", "#30A46C", "#3E63DD", "#8B8D98"


class Layout:
    def __init__(self) -> None:
        self.panels: list[dict] = []
        self.y = 0
        self.next_id = 1

    def row(self, title: str) -> None:
        self.panels.append({"type": "row", "title": title, "collapsed": False, "id": self._id(),
                            "gridPos": {"h": 1, "w": 24, "x": 0, "y": self.y}, "panels": []})
        self.y += 1

    def line(self, panels: list[tuple[dict, int]], h: int) -> None:
        x = 0
        for panel, w in panels:
            panel.update(id=self._id(), gridPos={"h": h, "w": w, "x": x, "y": self.y}, datasource=DS)
            self.panels.append(panel)
            x += w
        self.y += h

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id - 1


def target(expr: str, legend: str = "", instant: bool = False, fmt: str = "time_series", ref: str = "A") -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "instant": instant,
            "range": not instant, "format": fmt}


def thresholds(*steps: tuple[float | None, str]) -> dict:
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in steps]}


def stat(title: str, expr: str, unit: str = "none", steps=None, decimals: int | None = None,
         mappings: list | None = None, description: str = "") -> dict:
    return {
        "type": "stat", "title": title, "description": description, "targets": [target(expr)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "mappings": mappings or [],
                                     "thresholds": steps or thresholds((None, BLUE)), "color": {"mode": "thresholds"}},
                        "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "background", "graphMode": "area", "textMode": "value", "justifyMode": "center"},
    }


def timeseries(title: str, targets: list[dict], unit: str = "none", overrides: list | None = None,
               steps=None, threshold_style: str = "off", description: str = "", stack: bool = False,
               min_: float | None = None, max_: float | None = None) -> dict:
    return {
        "type": "timeseries", "title": title, "description": description, "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit, "min": min_, "max": max_, "color": {"mode": "palette-classic"},
                "thresholds": steps or thresholds((None, GREEN)),
                "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 12, "gradientMode": "opacity",
                           "showPoints": "never", "spanNulls": True, "lineInterpolation": "smooth",
                           "thresholdsStyle": {"mode": threshold_style},
                           "stacking": {"mode": "normal" if stack else "none", "group": "A"}},
            },
            "overrides": overrides or [],
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def override(name: str, **props) -> dict:
    mapping = {"color": lambda c: {"id": "color", "value": {"mode": "fixed", "fixedColor": c}},
               "dash": lambda _: {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}},
               "fill": lambda f: {"id": "custom.fillOpacity", "value": f},
               "width": lambda w: {"id": "custom.lineWidth", "value": w}}
    return {"matcher": {"id": "byName", "options": name},
            "properties": [mapping[k](v) for k, v in props.items()]}


def build_dashboard(t: MonitoringTarget, bound: float | None = None, reference: float | None = None) -> dict:
    p, m = t.project, t.mapping.primary_metric
    sel = f'project="{p}"'
    req = f'mlmon_http_requests_total{{{sel},route="/predict"}}'
    higher = m in HIGHER_IS_BETTER
    L = Layout()

    # Couleur de la métrique : rouge au-delà de la limite de variabilité (calculée sur la référence)
    if bound is not None:
        perf_steps = thresholds((None, RED), (bound, GREEN)) if higher else thresholds((None, GREEN), (bound, RED))
    else:
        perf_steps = thresholds((None, BLUE))
    severity_map = [{"type": "value", "options": {
        "0": {"text": "OK", "color": GREEN, "index": 0}, "1": {"text": "INFO", "color": BLUE, "index": 1},
        "2": {"text": "WARNING", "color": ORANGE, "index": 2}, "3": {"text": "CRITICAL", "color": RED, "index": 3}}}]

    # -- Vue d'ensemble ------------------------------------------------------------
    L.row("Vue d'ensemble")
    L.line([
        (stat("État de la dernière fenêtre", f"mlmon_window_severity{{{sel}}}", mappings=severity_map,
              steps=thresholds((None, GREEN), (1, BLUE), (2, ORANGE), (3, RED)),
              description="Sévérité calculée par le moteur d'analyse (étape 1)."), 4),
        (stat(f"{m} estimé (sans labels)", f"mlmon_performance_estimated{{{sel},metric=\"{m}\"}}", decimals=4,
              steps=perf_steps, description="CBPE (classification) ou DLE (régression)."), 4),
        (stat(f"{m} mesuré", f"mlmon_performance_realized{{{sel},metric=\"{m}\"}}", decimals=4, steps=perf_steps,
              description="Calculé quand la vérité terrain est arrivée via /feedback."), 4),
        # Un pourcentage trompe (3 features sur 20 = 15 %, « vert », même avec un PSI de 8) :
        # on affiche le NOMBRE de features en drift MAJEUR (PSI >= 0.25), qui est ce qui compte.
        (stat("Features en drift majeur", f"count(mlmon_feature_psi{{{sel}}} >= 0.25) or vector(0)",
              steps=thresholds((None, GREEN), (1, ORANGE), (3, RED)),
              description="Nombre de features avec PSI >= 0.25 dans la dernière fenêtre."), 4),
        # `or vector(0)` : sans aucune erreur, la série n'existe pas -> « No data » au lieu de 0 %.
        (stat("Erreurs /predict", f'(sum(rate(mlmon_http_requests_total{{{sel},route="/predict",status!~"2.."}}[2m])) '
              f"or vector(0)) / clamp_min(sum(rate({req}[2m])), 1e-9)", unit="percentunit", decimals=1,
              steps=thresholds((None, GREEN), (0.01, ORANGE), (0.05, RED))), 4),
        (stat("Réveils de l'agent", f"mlmon_agent_triggers_total{{{sel}}}",
              steps=thresholds((None, GREEN), (1, ORANGE)),
              description="Fenêtres dont les signaux ont justifié un appel au LLM."), 4),
    ], h=5)

    # -- Modèle ------------------------------------------------------------------------
    L.row("Modèle : performance estimée vs mesurée")
    perf = timeseries(
        f"{m} : estimé, mesuré, habituel et limite",
        [target(f'mlmon_performance_estimated{{{sel},metric="{m}"}}', "estimé (sans labels)", ref="A"),
         target(f'mlmon_performance_realized{{{sel},metric="{m}"}}', "mesuré (labels)", ref="B"),
         target(f'mlmon_performance_reference{{{sel},metric="{m}"}}', "habituel (référence)", ref="C"),
         target(f'mlmon_performance_bound{{{sel},metric="{m}"}}', "limite (± 3σ)", ref="D")],
        overrides=[override("estimé (sans labels)", color=BLUE, width=3),
                   override("mesuré (labels)", color=GREEN, width=3),
                   override("habituel (référence)", color=GREY, dash=True, fill=0, width=1),
                   override("limite (± 3σ)", color=RED, dash=True, fill=0, width=2)],
        description="L'estimation arrive tout de suite ; la mesure arrive quand les labels arrivent. "
                    "Un écart durable entre les deux signale un concept drift.")
    alerts = {
        "type": "table", "title": "Alertes Prometheus actives",
        "targets": [target(f'ALERTS{{alertstate="firing",{sel}}}', instant=True, fmt="table")],
        "transformations": [{"id": "organize", "options": {
            "excludeByName": {"Time": True, "Value": True, "__name__": True, "alertstate": True, "project": True,
                              "job": True, "instance": True},
            "indexByName": {"severity": 0, "alertname": 1, "component": 2, "feature": 3}}}],
        "fieldConfig": {"defaults": {"custom": {"align": "left"}}, "overrides": [
            {"matcher": {"id": "byName", "options": "severity"},
             "properties": [{"id": "custom.cellOptions", "value": {"type": "color-background"}},
                            {"id": "mappings", "value": [{"type": "value", "options": {
                                "critical": {"color": RED, "index": 0}, "warning": {"color": ORANGE, "index": 1},
                                "info": {"color": BLUE, "index": 2}}}]}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }
    L.line([(perf, 16), (alerts, 8)], h=9)

    score = {
        "type": "heatmap", "title": "Distribution du score du modèle",
        "description": "Si la forme change, le modèle voit des cas différents (prediction drift).",
        "targets": [target(f"sum by (le) (increase(mlmon_prediction_score_bucket{{{sel}}}[1m]))", "{{le}}",
                           fmt="heatmap")],
        "options": {"calculate": False, "yAxis": {"axisPlacement": "left"}, "color": {"scheme": "Turbo", "mode": "scheme"},
                    "cellGap": 1, "tooltip": {"mode": "single"}},
    }
    classes = timeseries("Prédictions par classe (/s)",
                         [target(f"sum by (predicted) (rate(mlmon_predictions_total{{{sel}}}[1m]))", "classe {{predicted}}")],
                         unit="reqps", stack=True)
    L.line([(score, 12), (classes, 12)], h=8)

    # -- Données -------------------------------------------------------------------------
    L.row("Données : drift et qualité")
    psi = timeseries("PSI par feature", [target(f"mlmon_feature_psi{{{sel}}}", "{{feature}}")],
                     steps=thresholds((None, "transparent"), (0.1, ORANGE), (0.25, RED)), threshold_style="line+area",
                     description="PSI < 0.1 stable · 0.1–0.25 modéré · > 0.25 majeur.", min_=0)
    drifted = {
        "type": "state-timeline", "title": "Features en drift (fenêtre par fenêtre)",
        "targets": [target(f"mlmon_feature_drifted{{{sel}}}", "{{feature}}")],
        "fieldConfig": {"defaults": {"color": {"mode": "thresholds"},
                                     "thresholds": thresholds((None, GREEN), (1, RED)),
                                     "mappings": [{"type": "value", "options": {"0": {"text": "stable", "color": GREEN},
                                                                                 "1": {"text": "drift", "color": RED}}}]},
                        "overrides": []},
        "options": {"showValue": "never", "rowHeight": 0.8, "mergeValues": True, "alignValue": "center",
                    "legend": {"showLegend": False}},
    }
    L.line([(psi, 12), (drifted, 12)], h=9)

    psi_now = {
        "type": "bargauge", "title": "PSI actuel",
        "targets": [target(f"sort_desc(mlmon_feature_psi{{{sel}}})", "{{feature}}", instant=True)],
        "fieldConfig": {"defaults": {"min": 0, "max": 0.5, "decimals": 3, "color": {"mode": "thresholds"},
                                     "thresholds": thresholds((None, GREEN), (0.1, ORANGE), (0.25, RED))},
                        "overrides": []},
        "options": {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
    }
    quality = {
        "type": "table", "title": "Problèmes de qualité (dernière fenêtre)",
        "targets": [target(f"mlmon_data_quality_issue{{{sel}}}", instant=True, fmt="table")],
        "transformations": [{"id": "organize", "options": {
            "excludeByName": {"Time": True, "__name__": True, "project": True, "job": True, "instance": True},
            "renameByName": {"Value": "taux", "feature": "feature", "kind": "problème", "severity": "sévérité"}}}],
        "fieldConfig": {"defaults": {"custom": {"align": "left"}},
                        "overrides": [{"matcher": {"id": "byName", "options": "taux"},
                                       "properties": [{"id": "unit", "value": "percentunit"},
                                                      {"id": "custom.cellOptions",
                                                       "value": {"type": "gauge", "mode": "gradient"}},
                                                      {"id": "max", "value": 1}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }
    L.line([(psi_now, 10), (quality, 14)], h=8)

    # -- Service ---------------------------------------------------------------------------
    L.row("Service : trafic, erreurs, latence")
    traffic = timeseries("Requêtes /predict par statut", [target(f"sum by (status) (rate({req}[1m]))", "{{status}}")],
                         unit="reqps", stack=True)
    lat = f'sum by (le) (rate(mlmon_http_request_duration_seconds_bucket{{{sel},route="/predict"}}[1m]))'
    latency = timeseries("Latence", [target(f"histogram_quantile({q}, {lat})", f"p{int(q * 100)}", ref=r)
                                     for q, r in ((0.5, "A"), (0.95, "B"), (0.99, "C"))], unit="s")
    up = timeseries("Disponibilité de l'API du modèle", [target(f"mlmon_upstream_up{{{sel}}}", "upstream")],
                    min_=0, max_=1, steps=thresholds((None, RED), (1, GREEN)), threshold_style="area")
    L.line([(traffic, 8), (latency, 8), (up, 8)], h=8)

    return {
        "uid": f"mlmon-{p}"[:40], "title": f"Agent 4 · Monitoring · {p}", "tags": ["agent4", "mlops", p],
        "timezone": "browser", "schemaVersion": 39, "version": 1, "editable": True, "graphTooltip": 1,
        "refresh": "5s", "time": {"from": "now-30m", "to": "now"}, "panels": L.panels,
        "templating": {"list": []}, "annotations": {"list": []},
        "description": f"Généré par l'Agent 4. Métrique : {m} · seuil métier : {t.mapping.min_threshold}",
    }


def reference_band(t: MonitoringTarget) -> tuple[float | None, float | None]:
    """Limite et moyenne de la métrique sur la référence (pour colorer les seuils)."""
    ref = pd.read_parquet(t.reference_path)
    a = Analyzer(ref, t.mapping, AnalyzerConfig(window_size=t.window_size))
    if a.profile.metric_mean is None:
        return None, None
    k = a.config.performance.n_std * a.profile.metric_std
    bound = a.profile.metric_mean - k if t.mapping.primary_metric in HIGHER_IS_BETTER else a.profile.metric_mean + k
    return round(bound, 4), round(a.profile.metric_mean, 4)


def write_dashboard(t: MonitoringTarget, out_dir: Path) -> Path:
    bound, mean = reference_band(t)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{t.project}.json"
    path.write_text(json.dumps(build_dashboard(t, bound, mean), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
