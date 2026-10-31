"""
Superviseur : relie le moniteur (qui détecte) à l'agent (qui décide).

- construit le dossier d'enquête (Case) à partir de la base : fenêtres, alertes actives, décisions passées ;
- anti-rafale : une fenêtre de 200 prédictions arrive toutes les ~10 s. Sans garde-fou, l'agent
  rediagnostiquerait la même situation en boucle (coût, quotas, bruit). On ne relance l'agent que si
  la situation change (nouveaux types de signaux) ou après `every_windows` fenêtres ;
- l'agent tourne dans un thread à part : le monitoring continue pendant que le LLM réfléchit ;
- les décisions en attente d'un humain sont reprises via /monitoring/decisions/{id}/approve|reject.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pandas as pd

from ..agent.context import Case
from ..agent.graph import MonitoringAgent
from ..agent.llm import decider_and_judge
from ..agent.schemas import AgentRun
from ..schemas import WindowReport
from .store import Store
from .target import MonitoringTarget

log = logging.getLogger("monitoring_agent.supervisor")


class Supervisor:
    def __init__(self, target: MonitoringTarget, store: Store, agent: MonitoringAgent | None = None,
                 every_windows: int = 5, history: int = 20):
        self.target, self.store = target, store
        self.every_windows, self.history = every_windows, history
        if agent is None:
            decider, judge = decider_and_judge()
            from ..memory.store import default_memory
            agent = MonitoringAgent(decider, judge, out_dir=target.dir / "actions", memory=default_memory())
        agent.on_update = self._save
        self.agent = agent
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent4")
        self.running: Future | None = None
        self.busy_window: int | None = None
        # Réveil arrivé pendant que l'agent travaillait : on garde le plus récent et on le rejoue après.
        # (Sans ça, des labels qui révèlent un concept drift pendant une enquête étaient perdus.)
        self.pending: WindowReport | None = None
        self._lock = threading.Lock()

    # -- dossier d'enquête --------------------------------------------------------------

    def build_case(self, upto: int | None = None, detailed: bool = False) -> Case:
        reports = [WindowReport.model_validate(r) for r in self.store.window_reports(self.history + 50)]
        if upto is not None:
            reports = [r for r in reports if r.window.index <= upto]
        if not reports:
            raise RuntimeError("aucune fenêtre analysée")
        return Case(project=self.target.project, mapping=self.target.mapping, history=reports[-self.history:],
                    alerts=self.store.firing_alerts(), decisions=self.store.decisions(20, detailed=detailed),
                    model_uri=self.target.model_uri, export_new_data=self.export_new_data)

    def export_new_data(self, path: str) -> str | None:
        """Données labellisées des fenêtres en drift : ce sur quoi l'Agent 2 pourra réentraîner."""
        frames = []
        w = self.target.window_size
        for r in self.store.window_reports(self.history):
            if r["dataset_drift"]["n_drifted"] > 0 and r["performance"] and r["performance"]["labels_available"]:
                i = r["window"]["index"]
                frames.append(self.store.labelled_frame(i * w + 1, (i + 1) * w, self.target.mapping.target))
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        cols = self.target.mapping.features + [self.target.mapping.target]
        df[[c for c in cols if c in df.columns]].to_parquet(path, index=False)
        return path

    # -- déclenchement -----------------------------------------------------------------------

    def should_run(self, report: WindowReport) -> tuple[bool, str]:
        if self.running and not self.running.done():
            return False, "l'agent travaille déjà"
        decisions = self.store.decisions(20)
        if any(d["status"] == "awaiting_human" for d in decisions):
            return False, "une décision attend un humain"
        kinds = sorted({s.kind for s in report.signals})
        last = decisions[-1] if decisions else None
        if last and set(kinds) <= set(last["signals"]) and report.window.index - last["window_index"] < self.every_windows:
            return False, f"même situation que la décision {last['id']} (fenêtre {last['window_index']})"
        return True, "nouvelle situation"

    def on_trigger(self, report: WindowReport) -> None:
        with self._lock:
            if self.running and not self.running.done():
                self.pending = report
                log.info("fenêtre %d : l'agent travaille, réveil mis en attente", report.window.index)
                return
            ok, why = self.should_run(report)
            if not ok:
                log.info("fenêtre %d : agent non relancé (%s)", report.window.index, why)
                return
            log.info("fenêtre %d : l'agent enquête (%s)", report.window.index, why)
            self.busy_window = report.window.index
            self.running = self.pool.submit(self._decide, report.window.index)
            self.running.add_done_callback(self._after)

    def _after(self, _future: Future) -> None:
        with self._lock:
            pending, self.pending, self.busy_window = self.pending, None, None
        if pending is not None:
            log.info("fenêtre %d : réveil en attente rejoué", pending.window.index)
            self.on_trigger(pending)

    def decide_now(self) -> AgentRun:
        return self._decide(None)

    def _decide(self, upto: int | None) -> AgentRun:
        run = self.agent.decide(self.build_case(upto))
        p = run.policy
        log.info("décision %s : %s -> %s (%s) %s", run.id, run.diagnosis.root_cause if run.diagnosis else "?",
                 p.final_action if p else "?", run.status, run.outputs or "")
        return run

    def resume(self, decision_id: str, decision: str, by: str, comment: str) -> AgentRun:
        return self.agent.resume(decision_id, decision, by, comment)

    def _save(self, run: AgentRun) -> None:
        case = self.agent.cases.get(run.id)
        self.store.save_decision(run, sorted(case.signal_kinds()) if case else [])

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
