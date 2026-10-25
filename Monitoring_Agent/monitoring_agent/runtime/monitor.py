"""
Moniteur : transforme le flux de prédictions en fenêtres analysées.

À chaque passage (toutes les quelques secondes) :
1. dès que W nouvelles prédictions sont arrivées -> on analyse cette fenêtre (étape 1) ;
2. on revisite les fenêtres sans labels : si la vérité terrain est arrivée entre-temps,
   on recalcule la performance MESURÉE (elle remplace l'estimation) ;
3. on met à jour les métriques Prometheus et on garde le rapport en base ;
4. si une fenêtre doit réveiller l'agent, on appelle `on_trigger` (branché à l'étape 3).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from ..analysis import Analyzer
from ..config import AnalyzerConfig
from ..schemas import WindowReport
from .metrics import GatewayMetrics
from .store import Store
from .target import MonitoringTarget

log = logging.getLogger("monitoring_agent.monitor")


class Monitor:
    def __init__(self, target: MonitoringTarget, store: Store, metrics: GatewayMetrics,
                 on_trigger: Callable[[WindowReport], None] | None = None):
        self.target, self.store, self.metrics = target, store, metrics
        self.on_trigger = on_trigger or (lambda w: None)
        reference = pd.read_parquet(target.reference_path)
        self.analyzer = Analyzer(reference, target.mapping, AnalyzerConfig(window_size=target.window_size))
        self.last_report: WindowReport | None = None
        self.last_labelled_idx = -1   # la perf MESURÉE affichée est celle de la fenêtre labellisée la plus récente

    def _bounds(self, idx: int) -> tuple[int, int]:
        w = self.target.window_size
        return idx * w + 1, (idx + 1) * w

    def _analyze(self, idx: int) -> WindowReport:
        start, end = self._bounds(idx)
        frame = self.store.window_frame(start, end, self.target.mapping.target)
        report = self.analyzer.analyze_window(frame, idx)
        labels = bool(report.performance and report.performance.labels_available)
        self.store.save_window(idx, start, end, labels, report.severity, report.trigger_agent,
                               report.model_dump_json())
        return report

    def step(self) -> list[WindowReport]:
        """Un passage du moniteur. Renvoie les NOUVELLES fenêtres analysées."""
        new: list[WindowReport] = []
        complete = self.store.n_ok // self.target.window_size
        for idx in range(self.store.last_window_idx() + 1, complete):
            report = self._analyze(idx)
            new.append(report)
            self.metrics.windows.labels(self.target.project).inc()
            self.metrics.update_window(report)
            self.last_report = report
            log.info("fenêtre %d : %s%s", idx, report.severity, " -> agent" if report.trigger_agent else "")
            if report.trigger_agent:
                self.metrics.triggers.labels(self.target.project).inc()
                self.on_trigger(report)

            if report.performance and report.performance.labels_available:
                self.last_labelled_idx = max(self.last_labelled_idx, idx)

        # Labels arrivés en retard : la performance mesurée remplace l'estimation.
        for idx, start, end in sorted(self.store.windows_without_labels()):
            if self.store.labelled_share(start, end) >= 0.9:
                before = self.store.window_signal_kinds(idx)
                report = self._analyze(idx)
                perf = report.performance
                log.info("fenêtre %d : labels reçus, %s mesuré = %s", idx, perf.metric if perf else "-",
                         perf.realized if perf else None)
                if perf and perf.labels_available and idx > self.last_labelled_idx:
                    self.last_labelled_idx = idx
                    self.metrics.set_realized(perf)
                # Les labels peuvent RÉVÉLER un problème invisible sans eux (baisse mesurée, concept drift) :
                # c'est une situation nouvelle, l'agent doit être réveillé pour cette fenêtre.
                revealed = {s.kind for s in report.signals} - before
                if report.trigger_agent and revealed:
                    log.info("fenêtre %d : les labels révèlent %s -> agent", idx, ", ".join(sorted(revealed)))
                    self.metrics.triggers.labels(self.target.project).inc()
                    self.on_trigger(report)
        return new
