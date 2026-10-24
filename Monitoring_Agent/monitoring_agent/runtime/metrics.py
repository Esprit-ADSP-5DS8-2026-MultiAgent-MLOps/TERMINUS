"""
Métriques Prometheus exposées par la gateway sur /metrics.

Deux familles :
- SERVICE (mises à jour à chaque requête) : trafic, erreurs, latence, prédictions ;
- MODÈLE & DONNÉES (mises à jour à chaque fenêtre analysée) : drift, qualité, performance.

Prometheus les collecte toutes les 5 s ; les règles d'alerte et Grafana ne lisent que ça.
Préfixe commun `mlmon_` : facile à retrouver, ne collisionne avec rien.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from ..schemas import SEVERITY_RANK, WindowReport

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
SCORE_BUCKETS = tuple(round(i / 10, 1) for i in range(1, 11))


class GatewayMetrics:
    def __init__(self, project: str):
        self.project = project
        self.registry = r = CollectorRegistry()
        L = ["project"]

        # -- service --
        self.requests = Counter("mlmon_http_requests", "Requêtes reçues par la gateway",
                                L + ["route", "status"], registry=r)
        self.latency = Histogram("mlmon_http_request_duration_seconds", "Latence de bout en bout (gateway + modèle)",
                                 L + ["route"], buckets=LATENCY_BUCKETS, registry=r)
        self.upstream_up = Gauge("mlmon_upstream_up", "1 si l'API du modèle répond à /health", L, registry=r)
        self.predictions = Counter("mlmon_predictions", "Prédictions servies, par classe", L + ["predicted"], registry=r)
        self.score = Histogram("mlmon_prediction_score", "Distribution du score (proba) renvoyé par le modèle",
                               L, buckets=SCORE_BUCKETS, registry=r)
        self.labels = Counter("mlmon_labels_received", "Labels (vérité terrain) reçus via /feedback", L, registry=r)

        # -- modèle & données (dernière fenêtre analysée) --
        self.window_index = Gauge("mlmon_window_index", "Index de la dernière fenêtre analysée", L, registry=r)
        self.window_severity = Gauge("mlmon_window_severity", "Sévérité : 0 ok, 1 info, 2 warning, 3 critical", L, registry=r)
        self.windows = Counter("mlmon_windows_analyzed", "Fenêtres analysées", L, registry=r)
        self.triggers = Counter("mlmon_agent_triggers", "Fenêtres qui ont réveillé l'agent", L, registry=r)
        self.dataset_drift = Gauge("mlmon_dataset_drift_share", "Part des features en drift", L, registry=r)
        self.feature_drift = Gauge("mlmon_feature_drift_score", "Score de drift par feature",
                                   L + ["feature", "method"], registry=r)
        self.feature_drifted = Gauge("mlmon_feature_drifted", "1 si la feature est en drift", L + ["feature"], registry=r)
        self.feature_psi = Gauge("mlmon_feature_psi", "PSI par feature", L + ["feature"], registry=r)
        self.prediction_drifted = Gauge("mlmon_prediction_drifted", "1 si la distribution des prédictions a dérivé",
                                        L, registry=r)
        self.quality = Gauge("mlmon_data_quality_issue", "Taux observé pour chaque problème de qualité",
                             L + ["feature", "kind", "severity"], registry=r)
        self.quality_severity = Gauge("mlmon_data_quality_severity", "Pire sévérité qualité (0 à 3)", L, registry=r)
        self.perf_estimated = Gauge("mlmon_performance_estimated", "Performance estimée sans labels",
                                    L + ["metric", "method"], registry=r)
        self.perf_realized = Gauge("mlmon_performance_realized", "Performance mesurée avec les labels",
                                   L + ["metric"], registry=r)
        self.perf_reference = Gauge("mlmon_performance_reference", "Performance habituelle (référence)",
                                    L + ["metric"], registry=r)
        self.perf_bound = Gauge("mlmon_performance_bound", "Limite de la variabilité naturelle",
                                L + ["metric"], registry=r)

    # -- mises à jour --------------------------------------------------------------

    def observe_request(self, route: str, status: int, seconds: float) -> None:
        self.requests.labels(self.project, route, str(status)).inc()
        self.latency.labels(self.project, route).observe(seconds)

    def observe_prediction(self, prediction: float | None, proba: float | None) -> None:
        if prediction is not None:
            label = str(int(prediction)) if float(prediction).is_integer() else "value"
            self.predictions.labels(self.project, label).inc()
        if proba is not None:
            self.score.labels(self.project).observe(proba)

    def update_window(self, w: WindowReport) -> None:
        p = self.project
        self.window_index.labels(p).set(w.window.index)
        self.window_severity.labels(p).set(SEVERITY_RANK[w.severity])
        self.dataset_drift.labels(p).set(w.dataset_drift.share_drifted)
        self.feature_drift.clear()  # la méthode peut changer d'une fenêtre à l'autre (KS <-> Wasserstein)
        for d in w.feature_drift:
            self.feature_drift.labels(p, d.feature, d.method).set(d.score)
            self.feature_drifted.labels(p, d.feature).set(int(d.drifted))
            self.feature_psi.labels(p, d.feature).set(d.psi)
        self.prediction_drifted.labels(p).set(int(bool(w.prediction_drift and w.prediction_drift.drifted)))
        self.quality.clear()        # un problème résolu doit disparaître
        for q in w.data_quality:
            self.quality.labels(p, q.feature, q.kind, q.severity).set(q.value)
        self.quality_severity.labels(p).set(max((SEVERITY_RANK[q.severity] for q in w.data_quality), default=0))
        self.update_performance(w)

    def update_performance(self, w: WindowReport) -> None:
        perf, p = w.performance, self.project
        if perf is None:
            return
        m = perf.metric
        if perf.reference_mean is not None:
            self.perf_reference.labels(p, m).set(perf.reference_mean)
        if perf.bound is not None:
            self.perf_bound.labels(p, m).set(perf.bound)
        if perf.estimated and m in perf.estimated:
            self.perf_estimated.labels(p, m, perf.estimation_method or "none").set(perf.estimated[m])
        self.set_realized(perf)

    def set_realized(self, perf) -> None:
        if perf is not None and perf.realized and perf.metric in perf.realized:
            self.perf_realized.labels(self.project, perf.metric).set(perf.realized[perf.metric])

    def render(self) -> bytes:
        return generate_latest(self.registry)
