"""Trace des étapes d'un agent : phases (contrôles statiques, environnement, exécution, LLM, politique) + appels d'outils.
Sert au rapport (champ `steps`), à `--steps` et à la progression en direct. Aucune décision n'en dépend."""
from __future__ import annotations

import time
from contextlib import contextmanager


class Trace:
    def __init__(self, progress=None, agent="validator"):
        self.t0 = time.time()
        self.progress, self.agent = progress, agent
        self.phases: list[dict] = []

    def _emit(self, rec):
        fn = getattr(self.progress, "on_step", None)
        if fn:
            try:
                fn(self.agent, rec["name"], rec.get("detail", ""), rec["status"], rec["duration_s"])
            except Exception:
                pass

    @contextmanager
    def phase(self, name: str, detail: str = ""):
        rec = {"kind": "phase", "name": name, "detail": detail, "status": "ok", "ts": time.time(), "duration_s": 0.0}
        try:
            yield rec                                     # le corps peut écrire rec["detail"] / rec["status"]
        except BaseException as e:
            rec["status"] = f"erreur : {type(e).__name__}"
            raise
        finally:
            rec["duration_s"] = round(time.time() - rec["ts"], 2)
            self.phases.append(rec)
            self._emit(rec)

    def event(self, name: str, detail: str = "", status: str = "ok"):
        rec = {"kind": "phase", "name": name, "detail": detail, "status": status, "ts": time.time(), "duration_s": 0.0}
        self.phases.append(rec)
        self._emit(rec)

    def export(self, toolbox=None) -> list[dict]:
        rows = list(self.phases) + list(getattr(toolbox, "trace", []) or [])
        rows.sort(key=lambda r: r.get("ts", 0))
        out = []
        for n, r in enumerate(rows, 1):
            out.append({"n": n, "kind": r["kind"], "name": r["name"], "detail": str(r.get("detail", ""))[:200],
                        "status": r.get("status", "ok"), "t": round(r.get("ts", self.t0) - self.t0, 2),
                        "duration_s": r.get("duration_s", 0.0)})
        return out
