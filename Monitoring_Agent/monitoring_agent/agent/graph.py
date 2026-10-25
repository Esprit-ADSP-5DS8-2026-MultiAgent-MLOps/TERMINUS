"""
Le graphe de décision de l'Agent 4 (LangGraph).

    prepare ─► investigate ─► diagnose ×3 ─► verify ──(preuves fausses, 1 fois)──► diagnose
                                                │
                                                ▼
                                              judge ──(« revise », 1 fois)──► diagnose
                                                │
                                                ▼
                                             policy ──auto──► act ─► fin
                                                │
                                                └─human─► await_human (interrupt) ─approve─► act
                                                                                 └─reject──► fin

- investigate : le LLM appelle des outils en lecture seule pour enquêter (ReAct borné) ;
- diagnose    : N diagnostics indépendants, vote majoritaire = self-consistency ;
- verify      : chaque chiffre cité est recontrôlé (déterministe) ; sinon une révision ;
- judge       : un modèle d'une AUTRE famille note le diagnostic sur une grille ;
- policy      : règles en code, qui substituent une action plus sûre et décident auto / humain ;
- await_human : pause LangGraph (interrupt) ; reprise avec la décision de l'humain.

Sans LLM configuré, `diagnose` utilise le décideur à règles (rules.py) : l'agent reste opérationnel.
"""

from __future__ import annotations

import operator
import time
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..config import AgentConfig
from . import prompts
from .actions import execute
from .context import Case, call_tool, case_facts, case_summary, compact, facts_block, tool_facts, tool_specs
from .guards import evaluate, verify
from .llm import ChatModel, LLMError, structured
from .rules import diagnose_by_rules
from .schemas import AgentRun, Diagnosis, JudgeVerdict, PolicyDecision, TraceStep, Verification


class AgentState(TypedDict, total=False):
    run_id: str
    facts: dict[str, float]
    notes: str
    observations: list[dict]
    samples: list[dict]
    diagnosis: dict
    agreement: float
    verification: dict
    verdicts: Annotated[list[dict], operator.add]
    critique: str
    revisions: int
    policy: dict
    status: str
    outputs: dict
    human: dict
    decider: str
    error: str
    trace: Annotated[list[dict], operator.add]


def _step(node: str, t0: float, summary: str, **detail) -> dict:
    return {"trace": [TraceStep(node=node, ms=round((time.perf_counter() - t0) * 1000, 1),
                                summary=summary, detail=detail).model_dump()]}


def vote(samples: list[Diagnosis]) -> tuple[Diagnosis, float]:
    """Action majoritaire ; parmi ses partisans, le diagnostic le plus confiant."""
    counts = Counter(s.action for s in samples)
    action, n = counts.most_common(1)[0]
    chosen = max((s for s in samples if s.action == action), key=lambda s: s.confidence)
    return chosen, n / len(samples)


class MonitoringAgent:
    def __init__(self, decider: ChatModel | None, judge: ChatModel | None, out_dir: Path,
                 config: AgentConfig | None = None, on_update: Callable[[AgentRun], None] | None = None):
        self.decider, self.judge = decider, judge
        self.out_dir = Path(out_dir)
        self.config = config or AgentConfig()
        self.on_update = on_update or (lambda run: None)
        self.cases: dict[str, Case] = {}
        self.graph = self._build().compile(checkpointer=MemorySaver())

    # -- API ----------------------------------------------------------------------------

    def decide(self, case: Case) -> AgentRun:
        run_id = f"D{case.current.window.index:04d}-{uuid.uuid4().hex[:6]}"
        self.cases[run_id] = case
        cfg = {"configurable": {"thread_id": run_id}}
        try:
            self.graph.invoke({"run_id": run_id, "revisions": 0, "observations": []}, cfg)
        except Exception as exc:  # noqa: BLE001 — l'agent ne doit jamais faire tomber la gateway
            return self._publish(run_id, error=f"{type(exc).__name__}: {exc}")
        return self._publish(run_id)

    def resume(self, run_id: str, decision: str, by: str = "humain", comment: str = "") -> AgentRun:
        if run_id not in self.cases:
            raise KeyError(f"décision inconnue : {run_id}")
        cfg = {"configurable": {"thread_id": run_id}}
        if not self.graph.get_state(cfg).interrupts:
            raise RuntimeError(f"la décision {run_id} n'attend pas d'humain")
        self.graph.invoke(Command(resume={"decision": decision, "by": by, "comment": comment}), cfg)
        return self._publish(run_id)

    def result(self, run_id: str, error: str | None = None) -> AgentRun:
        """État final (après invoke) : lu dans le checkpoint de LangGraph."""
        values = self.graph.get_state({"configurable": {"thread_id": run_id}}).values
        return self._assemble(run_id, values, error)

    def _assemble(self, run_id: str, st: dict, error: str | None = None) -> AgentRun:
        case = self.cases[run_id]
        status = st.get("status") or ("awaiting_human" if st.get("policy") and not st.get("outputs") else "failed")
        return AgentRun(
            id=run_id, project=case.project, window_index=case.current.window.index,
            decider=st.get("decider", ""), judge=self.judge.name if self.judge else "",
            samples=[Diagnosis(**x) for x in st.get("samples", [])], agreement=st.get("agreement", 0.0),
            diagnosis=Diagnosis(**st["diagnosis"]) if st.get("diagnosis") else None,
            verification=Verification(**st["verification"]) if st.get("verification") else None,
            verdicts=[JudgeVerdict(**v) for v in st.get("verdicts", [])],
            policy=PolicyDecision(**st["policy"]) if st.get("policy") else None,
            status="failed" if error else status, outputs=st.get("outputs", {}), human=st.get("human"),
            trace=[TraceStep(**t) for t in st.get("trace", [])], error=error or st.get("error"),
        )

    def _publish(self, run_id: str, error: str | None = None) -> AgentRun:
        run = self.result(run_id, error)
        self.on_update(run)
        return run

    # -- nœuds ------------------------------------------------------------------------------

    def _case(self, state: AgentState) -> Case:
        return self.cases[state["run_id"]]

    def _messages(self, system: str, case: Case, facts: dict, extra: str = "") -> list[dict]:
        user = f"{case_summary(case)}\n\nFACTS (clé = valeur) :\n{facts_block(facts)}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user + (f"\n\n{extra}" if extra else "")}]

    def prepare(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        facts = case_facts(case)
        return {"facts": facts, **_step("prepare", t0, f"{len(facts)} faits calculés, "
                                        f"{len(case.current.signals)} signaux", signals=[s.kind for s in case.current.signals])}

    def investigate(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        if self.decider is None:
            return _step("investigate", t0, "aucun LLM configuré : enquête sautée")
        case, facts = self._case(state), dict(state["facts"])
        messages = self._messages(prompts.INVESTIGATOR, case, facts)
        observations, notes = [], ""
        try:
            for _ in range(self.config.max_tool_calls):
                resp = self.decider.chat(messages, tools=tool_specs(), temperature=0.0)
                if not resp.tool_calls:
                    notes = resp.text.strip()
                    break
                messages.append(resp.raw_message)
                for call in resp.tool_calls:
                    result = call_tool(case, call.name, call.arguments)
                    facts.update(tool_facts(call.name, call.arguments, result))
                    observations.append({"tool": call.name, "args": call.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": compact(result)})
        except LLMError as exc:
            return {"facts": facts, "observations": observations,
                    **_step("investigate", t0, f"enquête interrompue : {exc}"[:200], error=str(exc))}
        calls = ", ".join(f"{o['tool']}({', '.join(f'{k}={v}' for k, v in o['args'].items())})" for o in observations)
        return {"facts": facts, "observations": observations, "notes": notes,
                **_step("investigate", t0, f"{len(observations)} appels d'outils : {calls or 'aucun'}",
                        notes=notes, tools=[{"tool": o["tool"], "args": o["args"]} for o in observations])}

    def diagnose(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        if self.decider is None:
            d = diagnose_by_rules(case)
            return {"samples": [d.model_dump()], "diagnosis": d.model_dump(), "agreement": 1.0, "decider": "rules",
                    **_step("diagnose", t0, f"décideur à règles : {d.root_cause} -> {d.action}")}
        extra = f"NOTE D'ENQUÊTE :\n{state.get('notes') or '(aucune)'}"
        if state.get("critique"):
            extra += f"\n\nTA PROPOSITION PRÉCÉDENTE A ÉTÉ REFUSÉE. Corrige-la :\n{state['critique']}"
        messages = self._messages(prompts.DIAGNOSER, case, state["facts"], extra)
        samples, errors = [], []
        for i in range(self.config.samples):
            temp = 0.0 if i == 0 else self.config.sample_temperature
            try:
                samples.append(structured(self.decider, messages, Diagnosis, temperature=temp))
            except LLMError as exc:
                errors.append(str(exc)[:200])
        if not samples:   # tous les appels ont échoué : repli sur les règles, l'agent continue
            d = diagnose_by_rules(case)
            return {"samples": [d.model_dump()], "diagnosis": d.model_dump(), "agreement": 1.0,
                    "decider": "rules (repli : LLM indisponible)", "error": "; ".join(errors),
                    **_step("diagnose", t0, "LLM indisponible : repli sur le décideur à règles", errors=errors)}
        chosen, agreement = vote(samples)
        name = getattr(self.decider, "last_used", "") or self.decider.name
        votes = dict(Counter(s.action for s in samples))
        return {"samples": [s.model_dump() for s in samples], "diagnosis": chosen.model_dump(),
                "agreement": agreement, "decider": name, "critique": "",
                **_step("diagnose", t0, f"{len(samples)} diagnostics, votes {votes} -> {chosen.root_cause} / "
                        f"{chosen.action} (accord {agreement:.0%})", votes=votes, errors=errors)}

    def verify_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        v = verify(Diagnosis(**state["diagnosis"]), state["facts"], case.mapping.features)
        update: dict[str, Any] = {"verification": v.model_dump()}
        if not v.ok and state.get("revisions", 0) < self.config.max_revisions and self.decider is not None:
            update.update(revisions=state.get("revisions", 0) + 1,
                          critique="Le vérificateur a trouvé des preuves fausses : " + "; ".join(v.problems))
        ok = sum(c.ok for c in v.checks)
        return {**update, **_step("verify", t0, f"{ok}/{len(v.checks)} preuves exactes"
                                  + (f" · problèmes : {'; '.join(v.problems)}" if v.problems else ""), problems=v.problems)}

    def judge_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        if self.judge is None:
            return _step("judge", t0, "aucun juge configuré")
        case = self._case(state)
        extra = (f"PROPOSITION À ÉVALUER :\n{compact(state['diagnosis'])}\n\n"
                 f"VÉRIFICATION AUTOMATIQUE DES PREUVES :\n{compact(state.get('verification'))}")
        try:
            v = structured(self.judge, self._messages(prompts.JUDGE, case, state["facts"], extra), JudgeVerdict)
        except LLMError as exc:
            return _step("judge", t0, f"juge indisponible : {exc}"[:200])
        update: dict[str, Any] = {"verdicts": [v.model_dump()]}
        if v.verdict == "revise" and state.get("revisions", 0) < self.config.max_revisions and self.decider is not None:
            update.update(revisions=state.get("revisions", 0) + 1, critique=f"Le juge demande une révision : {v.critique}")
        return {**update, **_step("judge", t0, f"{v.verdict} · {v.scores} · {v.critique[:160]}")}

    def policy_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        verdicts = state.get("verdicts") or []
        verdict = JudgeVerdict(**verdicts[-1]) if verdicts else None
        p = evaluate(Diagnosis(**state["diagnosis"]), case, Verification(**state["verification"]), verdict,
                     state.get("agreement", 0.0), self.config.policy)
        failed = [r.rule for r in p.rules if not r.passed]
        return {"policy": p.model_dump(), **_step(
            "policy", t0, f"{p.proposed_action} -> {p.final_action} · mode {p.mode}"
            + (f" · règles non satisfaites : {', '.join(failed)}" if failed else ""), reasons=p.reasons)}

    def await_human(self, state: AgentState) -> dict:
        p = state["policy"]
        answer = interrupt({"run_id": state["run_id"], "question": f"Exécuter « {p['final_action']} » ?",
                            "reasons": p["reasons"], "diagnosis": state["diagnosis"]})
        t0 = time.perf_counter()
        human = {"decision": answer.get("decision"), "by": answer.get("by", "humain"), "comment": answer.get("comment", "")}
        status = None if human["decision"] == "approve" else "rejected_by_human"
        upd: dict[str, Any] = {"human": human, **_step("await_human", t0, f"{human['decision']} par {human['by']}")}
        if status:
            upd["status"] = status
        return upd

    def act(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        # L'état reçu par le nœud, PAS le checkpoint : LangGraph écrit ses checkpoints en asynchrone,
        # relire le checkpoint ici pourrait donner une version en retard d'une étape.
        run = self._assemble(state["run_id"], dict(state))
        action = state["policy"]["final_action"]
        outputs = execute(action, run, case, self.out_dir)
        return {"status": "executed", "outputs": outputs,
                **_step("act", t0, f"{action} exécuté" + (f" -> {', '.join(outputs)}" if outputs else ""))}

    # -- graphe -------------------------------------------------------------------------------

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        for name, fn in (("prepare", self.prepare), ("investigate", self.investigate), ("diagnose", self.diagnose),
                         ("verify", self.verify_node), ("judge", self.judge_node), ("policy", self.policy_node),
                         ("await_human", self.await_human), ("act", self.act)):
            g.add_node(name, fn)
        g.add_edge(START, "prepare")
        g.add_edge("prepare", "investigate")
        g.add_edge("investigate", "diagnose")
        g.add_edge("diagnose", "verify")
        g.add_conditional_edges("verify", lambda s: "diagnose" if s.get("critique") else "judge", ["diagnose", "judge"])
        g.add_conditional_edges("judge", lambda s: "diagnose" if s.get("critique") else "policy", ["diagnose", "policy"])
        g.add_conditional_edges("policy", lambda s: "act" if s["policy"]["mode"] == "auto" else "await_human",
                                ["act", "await_human"])
        g.add_conditional_edges("await_human", lambda s: END if s.get("status") == "rejected_by_human" else "act",
                                ["act", END])
        g.add_edge("act", END)
        return g
