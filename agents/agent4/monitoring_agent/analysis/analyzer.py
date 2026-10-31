"""
Analyzer : le cœur déterministe de l'Agent 4.

    référence ──fit──► ReferenceProfile
    production ──fenêtres──► pour chaque fenêtre : drift, qualité, performance ──► signaux

Les signaux décident si on réveille le LLM (trigger_agent). Le LLM n'est jamais appelé
pour une fenêtre calme : c'est moins cher, plus rapide, et chaque appel a une raison.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from ..config import AnalyzerConfig
from ..schemas import (
    HIGHER_IS_BETTER,
    SEVERITY_RANK,
    ColumnMapping,
    MonitoringReport,
    PerformanceResult,
    QualityIssue,
    Signal,
    WindowInfo,
    WindowReport,
    max_severity,
)
from .drift import column_drift, dataset_drift
from .performance import CBPE, realized_metrics
from .quality import column_quality
from .reference import ReferenceProfile


def make_windows(df: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    """Fenêtres consécutives de `size` lignes. La dernière, si trop petite (< 50 %), est fusionnée."""
    n = len(df)
    if n == 0:
        return []
    bounds = list(range(0, n, size)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < size // 2:
        bounds.pop(-2)
    return [df.iloc[a:b] for a, b in itertools.pairwise(bounds)]


class Analyzer:
    def __init__(self, reference: pd.DataFrame, mapping: ColumnMapping, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()
        self.mapping = mapping
        self.profile = ReferenceProfile.fit(reference, mapping, self.config)

    # -- une fenêtre --------------------------------------------------------------

    def analyze_window(self, current: pd.DataFrame, index: int = 0) -> WindowReport:
        m, cfg, prof = self.mapping, self.config, self.profile
        info = self._window_info(current, index)

        present = [f for f in m.features if f in current.columns]
        drift = [column_drift(prof.columns[f], current[f], cfg.drift, n_tests=len(present)) for f in present]
        pred_col = m.prediction_proba or m.prediction
        pred_drift = column_drift(prof.prediction, current[pred_col], cfg.drift) if pred_col in current.columns else None
        ds = dataset_drift(drift, cfg.drift)

        quality: list[QualityIssue] = []
        for f in m.features:
            quality += column_quality(prof.columns[f], current, cfg.quality)

        perf = self._performance(current)
        signals = self._signals(ds, drift, pred_drift, quality, perf)
        severity = max_severity([s.severity for s in signals])
        return WindowReport(
            window=info, feature_drift=drift, prediction_drift=pred_drift, dataset_drift=ds,
            data_quality=quality, performance=perf, signals=signals, severity=severity,
            trigger_agent=SEVERITY_RANK[severity] >= SEVERITY_RANK["warning"],
        )

    # -- toute la production --------------------------------------------------------

    def analyze(self, production: pd.DataFrame, project: str = "project") -> MonitoringReport:
        windows = [self.analyze_window(w, i) for i, w in enumerate(make_windows(production, self.config.window_size))]
        return MonitoringReport(
            project=project, mapping=self.mapping, reference_rows=self.profile.n_rows,
            window_size=self.config.window_size, windows=windows, summary=self._summary(windows),
        )

    # -- détails ----------------------------------------------------------------------

    def _window_info(self, df: pd.DataFrame, index: int) -> WindowInfo:
        ts = self.mapping.timestamp
        start = end = None
        if ts and ts in df.columns and len(df):
            start, end = str(df[ts].iloc[0]), str(df[ts].iloc[-1])
        return WindowInfo(index=index, n_rows=len(df), start=start, end=end)

    def _performance(self, df: pd.DataFrame) -> PerformanceResult | None:
        m, prof = self.mapping, self.profile
        if prof.metric_mean is None:  # référence sans labels : rien à comparer
            return None
        metric, higher = m.primary_metric, m.primary_metric in HIGHER_IS_BETTER
        bound = (prof.metric_mean - self.config.performance.n_std * prof.metric_std if higher
                 else prof.metric_mean + self.config.performance.n_std * prof.metric_std)

        def degraded(value: float | None) -> bool | None:
            if value is None or math.isnan(value):
                return None
            below_band = value < bound if higher else value > bound
            below_min = m.min_threshold is not None and (value < m.min_threshold if higher else value > m.min_threshold)
            return bool(below_band or below_min)

        estimated = method = None
        if isinstance(prof.estimator, CBPE) and m.prediction_proba in df.columns:
            estimated, method = prof.estimator.estimate(df[m.prediction_proba], df[m.prediction]), "cbpe"
        elif prof.estimator is not None and m.task == "regression":
            estimated, method = prof.estimator.estimate(df, df[m.prediction]), "dle"

        realized = None
        labelled = df[df[m.target].notna()] if m.target and m.target in df.columns else df.iloc[0:0]
        # On n'affirme une performance mesurée que si (presque) toute la fenêtre est labellisée.
        labels_available = len(df) > 0 and len(labelled) >= 0.9 * len(df)
        if labels_available:
            proba = labelled[m.prediction_proba] if m.prediction_proba else None
            realized = realized_metrics(m.task, labelled[m.target], labelled[m.prediction], proba)

        def r(d):
            return {k: round(v, 6) for k, v in d.items()} if d else None

        return PerformanceResult(
            metric=metric, labels_available=labels_available, realized=r(realized), estimated=r(estimated),
            estimation_method=method, reference_mean=round(prof.metric_mean, 6),
            reference_std=round(prof.metric_std, 6), bound=round(bound, 6), min_threshold=m.min_threshold,
            realized_degraded=degraded(realized.get(metric)) if realized else None,
            estimated_degraded=degraded(estimated.get(metric)) if estimated else None,
        )

    def _signals(self, ds, drift, pred_drift, quality, perf) -> list[Signal]:
        out: list[Signal] = []
        top = sorted((d for d in drift if d.drifted), key=lambda d: d.psi, reverse=True)
        if ds.drifted:
            out.append(Signal(kind="dataset_drift", severity="warning",
                              message=f"{ds.n_drifted}/{ds.n_features} features en drift",
                              evidence={"share_drifted": ds.share_drifted}))
        elif top:
            # Peu de features, mais un drift majeur (PSI >= 0.25) sur l'une d'elles mérite l'agent :
            # sur 20 features, 3 qui explosent ne font pas 50 %, mais peuvent suffire à tromper le modèle.
            major = [d for d in top if d.psi_level == "major"]
            out.append(Signal(kind="feature_drift", severity="warning" if major else "info",
                              message=("drift majeur sur " if major else "drift sur ") + ", ".join(d.feature for d in top),
                              evidence={d.feature: d.psi for d in top}))
        if pred_drift and pred_drift.drifted:
            out.append(Signal(kind="prediction_drift", severity="warning",
                              message=f"la distribution des prédictions a changé ({pred_drift.method}={pred_drift.score:.3f})",
                              evidence={"score": pred_drift.score, "psi": pred_drift.psi}))
        serious = [q for q in quality if SEVERITY_RANK[q.severity] >= SEVERITY_RANK["warning"]]
        if serious:
            out.append(Signal(kind="data_quality", severity=max_severity([q.severity for q in serious]),
                              message="; ".join(f"{q.feature}: {q.kind} {q.value:.1%}" for q in serious),
                              evidence={f"{q.feature}.{q.kind}": q.value for q in serious}))
        if perf:
            metric = perf.metric
            if perf.realized_degraded:
                out.append(Signal(kind="performance_drop_realized", severity="critical",
                                  message=f"{metric} mesuré = {perf.realized[metric]:.4f} (habituel {perf.reference_mean:.4f}, limite {perf.bound:.4f})",
                                  evidence={"realized": perf.realized[metric], "reference_mean": perf.reference_mean,
                                            "bound": perf.bound}))
                # L'estimation (CBPE/DLE) suppose que la relation features -> cible est stable.
                # Si la mesure chute et que l'estimation, elle, reste normale, c'est cette relation
                # qui a changé : concept drift. Réentraîner sur des données récentes est alors justifié.
                if perf.estimated_degraded is False:
                    out.append(Signal(kind="concept_drift_suspected", severity="warning",
                                      message=f"{metric} mesuré {perf.realized[metric]:.4f} mais estimé {perf.estimated[metric]:.4f} : "
                                              "la relation features -> cible a probablement changé",
                                      evidence={"realized": perf.realized[metric], "estimated": perf.estimated[metric]}))
            elif perf.estimated_degraded:
                out.append(Signal(kind="performance_drop_estimated", severity="warning",
                                  message=f"{metric} estimé ({perf.estimation_method}) = {perf.estimated[metric]:.4f} (habituel {perf.reference_mean:.4f}, limite {perf.bound:.4f})",
                                  evidence={"estimated": perf.estimated[metric], "reference_mean": perf.reference_mean,
                                            "bound": perf.bound}))
        return out

    @staticmethod
    def _summary(windows: list[WindowReport]) -> dict:
        def first(pred):
            return next((w.window.index for w in windows if pred(w)), None)

        drift_counts: dict[str, int] = {}
        for w in windows:
            for f in w.dataset_drift.drifted_features:
                drift_counts[f] = drift_counts.get(f, 0) + 1
        kinds = lambda w: {s.kind for s in w.signals}
        return {
            "n_windows": len(windows),
            "n_triggered": sum(w.trigger_agent for w in windows),
            "worst_severity": max_severity([w.severity for w in windows]),
            "first_drift_window": first(lambda w: w.dataset_drift.n_drifted > 0),
            "first_quality_issue_window": first(lambda w: "data_quality" in kinds(w)),
            "first_estimated_drop_window": first(lambda w: "performance_drop_estimated" in kinds(w)
                                                 or bool(w.performance and w.performance.estimated_degraded)),
            "first_realized_drop_window": first(lambda w: "performance_drop_realized" in kinds(w)),
            "most_drifted_features": [f for f, _ in sorted(drift_counts.items(), key=lambda x: -x[1])[:5]],
        }

