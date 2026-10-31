"""Progression en direct : distingue les trois agents et affiche les tokens (dont ceux en cache)."""
from __future__ import annotations

import sys

LABELS = {"validator": "VALIDATOR", "repair": "REPAIR   ", "judge": "JUDGE    "}


class NullProgress:
    def on_start(self, agent, detail=""): ...
    def on_llm(self, agent, step, usage, latency_s, model=""): ...
    def on_tool(self, agent, step, name, args=""): ...
    def on_end(self, agent, stats, failure=None): ...
    def on_step(self, agent, name, detail="", status="ok", duration_s=0.0): ...


class ConsoleProgress(NullProgress):
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.totals: dict[str, dict] = {}

    def _p(self, agent, text):
        print(f"[{LABELS.get(agent, agent.upper())}] {text}", file=self.stream, flush=True)

    def on_start(self, agent, detail=""):
        self.totals[agent] = {"prompt": 0, "completion": 0, "cached": 0}
        self._p(agent, f"démarrage {detail}".rstrip())

    def on_llm(self, agent, step, usage, latency_s, model=""):
        t = self.totals.setdefault(agent, {"prompt": 0, "completion": 0, "cached": 0})
        t["prompt"] += usage.get("prompt_tokens", 0)
        t["completion"] += usage.get("completion_tokens", 0)
        t["cached"] += usage.get("cached_tokens", 0)
        self._p(agent, f"étape {step:>2} · {model or 'LLM'} · entrée {usage.get('prompt_tokens', 0)} "
                       f"(dont {usage.get('cached_tokens', 0)} en cache) · sortie {usage.get('completion_tokens', 0)} · {latency_s:.1f}s")

    def on_tool(self, agent, step, name, args=""):
        self._p(agent, f"         ↳ outil {name} {args}"[:420])

    def on_step(self, agent, name, detail="", status="ok", duration_s=0.0):
        mark = "✓" if status == "ok" else "•" if status == "ignorée" else "!"
        self._p(agent, f"{mark} {name} · {duration_s:.1f}s" + (f" · {detail}" if detail else "") + ("" if status in ("ok", "ignorée") else f" [{status}]"))

    def on_end(self, agent, stats, failure=None):
        t = self.totals.get(agent, {})
        msg = (f"terminé · {stats.get('steps', 0)} étape(s) · {stats.get('total_tokens', 0)} tokens "
               f"(cache {t.get('cached', 0)})")
        if failure:
            msg += f" · ÉCHEC FERMÉ : {failure.kind.value} — {failure.message}"
        self._p(agent, msg)
