"""
Dossier d'enquête : ce que l'agent SAIT d'une situation, et les outils pour en savoir plus.

Deux idées :
1. Les FAITS : tous les chiffres calculés (drift, qualité, performance...) sont mis à plat sous
   des clés stables (`drift.age.psi`, `perf.realized.roc_auc`...). Le LLM ne peut citer QUE ces
   clés : le vérificateur contrôle ensuite que chaque chiffre cité existe et est exact.
2. Les OUTILS : fonctions en lecture seule que le LLM peut appeler pendant l'enquête
   (historique d'une feature, de la performance, détails qualité, alertes, décisions passées).
   Chaque résultat d'outil enrichit les faits citables.

Les valeurs venant des données (noms de catégories...) sont des DONNÉES, jamais des instructions :
elles passent dans les faits ou en JSON, jamais dans le texte des consignes (prompt injection).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..schemas import ColumnMapping, WindowReport

SIGNAL_KINDS_PERF = {"performance_drop_realized", "performance_drop_estimated", "concept_drift_suspected"}


@dataclass
class Case:
    project: str
    mapping: ColumnMapping
    history: list[WindowReport]                    # fenêtres, de la plus ancienne à la courante
    alerts: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)   # décisions passées de l'agent (résumés)
    model_uri: str = "unknown"
    export_new_data: Callable[[str], str | None] | None = None   # exporte les données labellisées récentes
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def current(self) -> WindowReport:
        return self.history[-1]

    def window(self, index: int | None) -> WindowReport | None:
        if index is None:
            return self.current
        return next((w for w in self.history if w.window.index == index), None)

    def signal_kinds(self) -> set[str]:
        return {s.kind for s in self.current.signals}

    def has_quality_problem(self) -> bool:
        return any(s.kind == "data_quality" and s.severity in ("warning", "critical") for s in self.current.signals)


# ---------------------------------------------------------------------------
# Faits
# ---------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, int | float):
        return float(v)
    return None


def flatten(prefix: str, obj: Any, out: dict[str, float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(f"{prefix}.{i}", v, out)
    else:
        n = _num(obj)
        if n is not None:
            out[prefix] = round(n, 6)


def window_facts(w: WindowReport) -> dict[str, float]:
    f: dict[str, float] = {"window.index": w.window.index, "window.n_rows": w.window.n_rows,
                           "dataset.share_drifted": w.dataset_drift.share_drifted,
                           "dataset.n_drifted": w.dataset_drift.n_drifted,
                           "dataset.n_features": w.dataset_drift.n_features}
    for d in w.feature_drift:
        f[f"drift.{d.feature}.psi"] = d.psi
        if d.drifted:
            f[f"drift.{d.feature}.score"] = d.score
            f[f"drift.{d.feature}.drifted"] = 1.0
    if w.prediction_drift:
        f["prediction_drift.psi"] = w.prediction_drift.psi
        f["prediction_drift.score"] = w.prediction_drift.score
        f["prediction_drift.drifted"] = float(w.prediction_drift.drifted)
    for q in w.data_quality:
        f[f"quality.{q.feature}.{q.kind}"] = q.value
    p = w.performance
    if p:
        m = p.metric
        f["perf.labels_available"] = float(p.labels_available)
        for key, val in (("reference_mean", p.reference_mean), ("bound", p.bound), ("min_threshold", p.min_threshold)):
            if val is not None:
                f[f"perf.{key}"] = val
        if p.estimated and m in p.estimated:
            f[f"perf.estimated.{m}"] = p.estimated[m]
        if p.realized and m in p.realized:
            f[f"perf.realized.{m}"] = p.realized[m]
    return {k: round(float(v), 6) for k, v in f.items()}


def case_facts(case: Case) -> dict[str, float]:
    facts = window_facts(case.current)
    triggered = [w for w in case.history[-10:] if w.trigger_agent]
    streak = 0
    for w in reversed(case.history):
        if w.dataset_drift.n_drifted == 0:
            break
        streak += 1
    facts["history.windows_analyzed"] = float(len(case.history))
    facts["history.triggered_last_10"] = float(len(triggered))
    facts["history.consecutive_drift_windows"] = float(streak)
    facts["alerts.firing"] = float(len(case.alerts))
    return facts


def case_summary(case: Case) -> str:
    """Résumé déterministe de la situation, pour le LLM (et pour l'humain dans la console)."""
    w, m = case.current, case.mapping
    threshold = f" (seuil métier {m.min_threshold:g})" if m.min_threshold is not None else ""
    lines = [(f"Projet : {case.project} · tâche {m.task} · métrique principale {m.primary_metric}"
             f"{threshold} · {len(m.features)} features."),
             f"Fenêtre courante : n°{w.window.index}, {w.window.n_rows} prédictions, sévérité {w.severity}.",
             "Signaux calculés par le moteur d'analyse :"]
    lines += [f"  - [{s.severity}] {s.kind} : {s.message}" for s in w.signals] or ["  - aucun"]
    if case.alerts:
        lines.append("Alertes Prometheus actives : " + ", ".join(sorted({a['alertname'] for a in case.alerts})))
    if case.decisions:
        last = case.decisions[-1]
        lines.append(f"Dernière décision de l'agent : {last.get('action')} (fenêtre {last.get('window_index')}, "
                     f"{last.get('created_at')}).")
    return "\n".join(lines)


def facts_block(facts: dict[str, float]) -> str:
    return "\n".join(f"{k} = {v:g}" for k, v in sorted(facts.items()))


# ---------------------------------------------------------------------------
# Outils (lecture seule)
# ---------------------------------------------------------------------------


def tool_feature_history(case: Case, feature: str, last_n: int = 8) -> dict:
    rows = {}
    for w in case.history[-last_n:]:
        d = next((x for x in w.feature_drift if x.feature == feature), None)
        if d:
            rows[f"w{w.window.index}"] = {"psi": d.psi, "drifted": d.drifted, "method": d.method, "score": d.score}
    cur = next((x for x in case.current.feature_drift if x.feature == feature), None)
    if cur is None:
        return {"error": f"feature inconnue : {feature}", "features": case.mapping.features}
    return {"feature": feature, "kind": cur.kind, "history": rows,
            "reference": cur.reference_summary, "current": cur.current_summary}


def tool_performance_history(case: Case, last_n: int = 8) -> dict:
    out = {}
    for w in case.history[-last_n:]:
        p = w.performance
        if p:
            out[f"w{w.window.index}"] = {"estimated": (p.estimated or {}).get(p.metric),
                                         "realized": (p.realized or {}).get(p.metric),
                                         "labels_available": p.labels_available}
    p = case.current.performance
    return {"metric": case.mapping.primary_metric, "reference_mean": p.reference_mean if p else None,
            "bound": p.bound if p else None, "windows": out}


def tool_quality_details(case: Case, window_index: int | None = None) -> dict:
    w = case.window(window_index)
    if w is None:
        return {"error": f"fenêtre {window_index} absente"}
    return {"window": w.window.index,
            "issues": [{"feature": q.feature, "kind": q.kind, "rate": q.value, "reference_rate": q.reference_value,
                        "severity": q.severity, "detail": q.detail} for q in w.data_quality]}


def tool_prediction_drift(case: Case) -> dict:
    d = case.current.prediction_drift
    if d is None:
        return {"available": False}
    return {"drifted": d.drifted, "psi": d.psi, "score": d.score, "method": d.method,
            "reference": d.reference_summary, "current": d.current_summary}


def tool_active_alerts(case: Case) -> dict:
    return {"alerts": [{"alertname": a.get("alertname"), "severity": a.get("severity"),
                        "feature": a.get("payload", {}).get("labels", {}).get("feature")} for a in case.alerts]}


def tool_previous_decisions(case: Case) -> dict:
    return {"decisions": case.decisions[-5:]}


TOOLS: dict[str, tuple[Callable[..., dict], str, dict]] = {
    "feature_history": (tool_feature_history, "Historique du drift d'une feature (PSI par fenêtre) + stats référence/courant.",
                        {"feature": {"type": "string"}, "last_n": {"type": "integer"}}),
    "performance_history": (tool_performance_history, "Performance estimée et mesurée sur les dernières fenêtres.",
                            {"last_n": {"type": "integer"}}),
    "quality_details": (tool_quality_details, "Détail des problèmes de qualité d'une fenêtre (courante par défaut).",
                        {"window_index": {"type": "integer"}}),
    "prediction_drift": (tool_prediction_drift, "Drift de la distribution des prédictions du modèle.", {}),
    "active_alerts": (tool_active_alerts, "Alertes Prometheus actuellement actives.", {}),
    "previous_decisions": (tool_previous_decisions, "Les dernières décisions prises par l'agent.", {}),
}
REQUIRED = {"feature_history": ["feature"]}


def tool_specs() -> list[dict]:
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": REQUIRED.get(name, [])}}}
        for name, (_, desc, props) in TOOLS.items()]


def call_tool(case: Case, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"outil inconnu : {name}", "tools": list(TOOLS)}
    fn, _, props = TOOLS[name]
    clean = {k: v for k, v in args.items() if k in props}
    try:
        return fn(case, **clean)
    except (TypeError, ValueError) as exc:
        return {"error": f"arguments invalides : {exc}"}


def tool_facts(name: str, args: dict, result: dict) -> dict[str, float]:
    """Les nombres renvoyés par un outil deviennent des faits citables : tool.<nom>[.<feature>].<chemin>."""
    prefix = f"tool.{name}" + (f".{args['feature']}" if "feature" in args else "")
    out: dict[str, float] = {}
    flatten(prefix, result, out)
    return out


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
