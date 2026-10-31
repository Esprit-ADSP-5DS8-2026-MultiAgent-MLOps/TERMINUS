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
- judge       : un modèle d'une autre famille note le diagnostic sur une grille ;
- policy      : règles en code, qui substituent une action plus sûre et décident auto / humain ;
- await_human : pause LangGraph (interrupt) ; reprise avec la décision de l'humain.

Sans LLM configuré, `diagnose` utilise le décideur à règles (rules.py) : l'agent reste opérationnel.
"""

from __future__ import annotations

import contextvars
import logging
import operator
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .. import observability as obs
from ..config import AgentConfig
from ..memory.store import format_similar
from . import notify, prompts
from .actions import execute
from .context import Case, call_tool, case_facts, case_summary, compact, facts_block, tool_facts, tool_specs
from .guards import evaluate, fix_claims, verify
from .llm import ChatModel, LLMError, _settings, structured, usage_summary
from .rules import diagnose_by_rules
from .schemas import AgentRun, Diagnosis, JudgeVerdict, PolicyDecision, TraceStep, Verification

log = logging.getLogger("monitoring_agent.agent")


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
    similar: str                 # cas passés retrouvés par la mémoire d'incidents (RAG), déjà formatés
    similar_ids: list[str]
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


def _traced(name: str, fn):
    """Chaque nœud = une observation Langfuse typée (retriever, guardrail, evaluator…), si Langfuse est actif."""
    def node(state):
        with obs.observation(name, as_type=obs.NODE_TYPES.get(name, "span")) as o:
            out = fn(state)
            steps = out.get("trace") or []
            obs.update(o, output=steps[-1]["summary"] if steps else None,
                       metadata={"ms": steps[-1]["ms"]} if steps else None)
            return out
    node.__name__ = fn.__name__
    return node


def build_messages(system: str, case: Case, facts: dict, notes: str | None = None, similar: str | None = None,
                   critique: str | None = None, extra: str | None = None) -> list[dict]:
    """Messages d'un appel : même format en production, en évaluation et dans le dataset de fine-tuning."""
    user = prompts.user_message(case_summary(case), facts_block(facts), notes=notes, similar=similar,
                                critique=critique, extra=extra)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def vote(samples: list[Diagnosis]) -> tuple[Diagnosis, float]:
    """Action majoritaire ; parmi ses partisans, le diagnostic le plus confiant."""
    counts = Counter(s.action for s in samples)
    action, n = counts.most_common(1)[0]
    chosen = max((s for s in samples if s.action == action), key=lambda s: s.confidence)
    return chosen, n / len(samples)


class MonitoringAgent:
    def __init__(self, decider: ChatModel | None, judge: ChatModel | None, out_dir: Path,
                 config: AgentConfig | None = None, on_update: Callable[[AgentRun], None] | None = None,
                 investigator: ChatModel | None = None, memory=None, learn: bool = True):
        self.decider, self.judge = decider, judge
        # Mémoire d'incidents (RAG hybride) : lue avant le diagnostic ; enrichie des décisions confirmées.
        self.memory, self.learn = memory, learn
        self._remembered: set[str] = set()
        self.trace_ids: dict[str, str | None] = {}     # décision -> trace Langfuse (scores, feedback humain)
        # Diagnostiqueur local (SLM fine-tuné) : l'enquête est confiée à un grand modèle, et le SLM reçoit
        # exactement le format sur lequel il a été entraîné (faits seuls).
        self.investigator = investigator or getattr(decider, "investigator", None) or decider
        self.facts_only = bool(getattr(decider, "local", False))
        self.out_dir = Path(out_dir)
        self.config = config or AgentConfig()
        always = [a.strip() for a in _settings().get("MONITORING_ALWAYS_HUMAN", "").split(",") if a.strip()]
        if always and not self.config.policy.always_human:
            self.config = replace(self.config, policy=replace(self.config.policy, always_human=tuple(always)))
        self.on_update = on_update or (lambda run: None)
        self.cases: dict[str, Case] = {}
        # Console et Slack peuvent répondre en même temps : la première réponse gagne, la seconde est refusée.
        self._resume_lock = threading.Lock()
        self.graph = self._build().compile(checkpointer=MemorySaver())

    # -- API ----------------------------------------------------------------------------

    def decide(self, case: Case) -> AgentRun:
        run_id = f"D{case.current.window.index:04d}-{uuid.uuid4().hex[:6]}"
        self.cases[run_id] = case
        cfg = {"configurable": {"thread_id": run_id}}
        meta = {"project": case.project, "window": case.current.window.index, "prompt_version": prompts.PROMPT_VERSION,
                "decider": getattr(self.decider, "name", "rules"), "signals": ",".join(sorted(case.signal_kinds()))}
        with obs.decision_trace(run_id, case.project, meta) as trace_id:
            self.trace_ids[run_id] = trace_id
            try:
                self.graph.invoke({"run_id": run_id, "revisions": 0, "observations": []}, cfg)
            except Exception as exc:  # noqa: BLE001 - l'agent ne doit jamais faire tomber la gateway
                run = self._publish(run_id, error=f"{type(exc).__name__}: {exc}")
            else:
                run = self._publish(run_id)
            obs.finish_decision(run)
        obs.score_run(trace_id, run)
        obs.flush()
        return run

    def resume(self, run_id: str, decision: str, by: str = "humain", comment: str = "") -> AgentRun:
        if run_id not in self.cases:
            raise KeyError(f"décision inconnue : {run_id}")
        cfg = {"configurable": {"thread_id": run_id}}
        with self._resume_lock:
            if not self.graph.get_state(cfg).interrupts:
                raise RuntimeError(f"la décision {run_id} n'attend pas d'humain")
            self.graph.invoke(Command(resume={"decision": decision, "by": by, "comment": comment}), cfg)
        obs.score_human(self.trace_ids.get(run_id), decision, comment)
        obs.flush()
        run = self._publish(run_id)
        if run.status == "rejected_by_human":
            # Une action approuvée est notifiée par le nœud act ; un rejet ne passe pas par act.
            self._notify({**self.graph.get_state(cfg).values, "run_id": run_id}, "rejected")
        return run

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
            trace_url=obs.trace_url(self.trace_ids.get(run_id)),
        )

    def _publish(self, run_id: str, error: str | None = None) -> AgentRun:
        run = self.result(run_id, error)
        if self.memory is not None and self.learn and run_id not in self._remembered \
                and run.status in ("executed", "rejected_by_human"):
            try:
                if self.memory.remember(run, self.cases[run_id]):
                    self._remembered.add(run_id)
            except Exception:
                log.warning("écriture en mémoire impossible", exc_info=True)
        self.on_update(run)
        return run

    # -- nœuds ------------------------------------------------------------------------------

    def _case(self, state: AgentState) -> Case:
        return self.cases[state["run_id"]]

    def _messages(self, system: str, case: Case, facts: dict, **parts: str | None) -> list[dict]:
        return build_messages(system, case, facts, **parts)

    def prepare(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        facts = case_facts(case)
        update: dict[str, Any] = {"facts": facts}
        retrieved = ""
        if self.memory is not None:
            try:
                hits = self.memory.search(case)
            except Exception:
                log.warning("mémoire d'incidents indisponible", exc_info=True)
                hits = []
            update.update(similar=format_similar(hits), similar_ids=[h["id"] for h in hits])
            retrieved = f", {len(hits)} cas similaires ({', '.join(h['id'] for h in hits) or 'aucun au-dessus du seuil'})"
            update["_retrieval"] = [{k: h[k] for k in ("id", "title", "root_cause", "action", "text_sim", "sig_sim", "rrf")}
                                    for h in hits]
        retrieval = update.pop("_retrieval", None)
        return {**update, **_step("prepare", t0, f"{len(facts)} faits calculés, {len(case.current.signals)} signaux"
                                  + retrieved, signals=[s.kind for s in case.current.signals], retrieval=retrieval)}

    def _toolbox(self, case: Case):
        """Outils de l'enquête : via un serveur MCP (découverte + appels standard), sinon appel direct."""
        if self.config.use_mcp:
            try:
                from .mcp_server import MCPToolbox, build_server
                return MCPToolbox(build_server(lambda: case)).__enter__(), "mcp"
            except Exception:
                log.warning("serveur MCP indisponible, appel direct des outils", exc_info=True)
        return None, "direct"

    def investigate(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        if self.investigator is None:
            return _step("investigate", t0, "aucun LLM configuré : enquête sautée")
        case, facts = self._case(state), dict(state["facts"])
        # Les cas similaires (RAG) vont à l'enquêteur et au juge dans tous les cas. Seul le Qwen local
        # ne les reçoit pas, parce qu'il a été entraîné sur les faits seuls.
        messages = self._messages(prompts.INVESTIGATOR, case, facts, similar=state.get("similar"))
        observations, notes, meter = [], "", []
        box, transport = self._toolbox(case)
        specs = box.specs if box else tool_specs()
        try:
            for _ in range(self.config.max_tool_calls):
                resp = self.investigator.chat(messages, tools=specs, temperature=0.0)
                meter.append(resp)
                if not resp.tool_calls:
                    notes = resp.text.strip()
                    break
                messages.append(resp.raw_message)
                for call in resp.tool_calls:
                    result = box.call(call.name, call.arguments) if box else call_tool(case, call.name, call.arguments)
                    facts.update(tool_facts(call.name, call.arguments, result))
                    observations.append({"tool": call.name, "args": call.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": compact(result)})
        except LLMError as exc:
            return {"facts": facts, "observations": observations,
                    **_step("investigate", t0, f"enquête interrompue : {exc}"[:200], error=str(exc),
                            usage=usage_summary(meter), transport=transport)}
        finally:
            if box:
                box.__exit__(None, None, None)
        calls = ", ".join(f"{o['tool']}({', '.join(f'{k}={v}' for k, v in o['args'].items())})" for o in observations)
        return {"facts": facts, "observations": observations, "notes": notes,
                **_step("investigate", t0, f"{len(observations)} appels d'outils ({transport}) : {calls or 'aucun'}",
                        notes=notes, tools=[{"tool": o["tool"], "args": o["args"]} for o in observations],
                        usage=usage_summary(meter), transport=transport, tools_available=len(specs))}

    def diagnose(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        if self.decider is None:
            d = diagnose_by_rules(case)
            return {"samples": [d.model_dump()], "diagnosis": d.model_dump(), "agreement": 1.0, "decider": "rules",
                    **_step("diagnose", t0, f"décideur à règles : {d.root_cause} -> {d.action}")}
        context = {} if self.facts_only else {"notes": state.get("notes"), "similar": state.get("similar")}
        messages = self._messages(prompts.DIAGNOSER, case, state["facts"], critique=state.get("critique"), **context)
        # Les N diagnostics sont indépendants : on les lance en parallèle (latence ÷ N).
        meter: list = []

        def one(i: int):
            temp = 0.0 if i == 0 else self.config.sample_temperature
            try:
                return structured(self.decider, messages, Diagnosis, temperature=temp, meter=meter), None
            except LLMError as exc:
                return None, str(exc)[:200]

        # Un contexte copié par tirage : les générations Langfuse restent rattachées à la trace de la décision.
        contexts = [contextvars.copy_context() for _ in range(self.config.samples)]
        with ThreadPoolExecutor(max_workers=self.config.samples) as pool:
            results = list(pool.map(lambda i: contexts[i].run(one, i), range(self.config.samples)))
        samples = [d for d, _ in results if d is not None]
        errors = [e for _, e in results if e]
        if not samples:   # tous les appels ont échoué : repli sur les règles, l'agent continue
            d = diagnose_by_rules(case)
            return {"samples": [d.model_dump()], "diagnosis": d.model_dump(), "agreement": 1.0,
                    "decider": "rules (repli : LLM indisponible)", "error": "; ".join(errors),
                    **_step("diagnose", t0, "LLM indisponible : repli sur le décideur à règles", errors=errors,
                            usage=usage_summary(meter))}
        chosen, agreement = vote(samples)
        name = getattr(self.decider, "last_used", "") or self.decider.name
        votes = dict(Counter(s.action for s in samples))
        return {"samples": [s.model_dump() for s in samples], "diagnosis": chosen.model_dump(),
                "agreement": agreement, "decider": name, "critique": "",
                **_step("diagnose", t0, f"{len(samples)} diagnostics, votes {votes} -> {chosen.root_cause} / "
                        f"{chosen.action} (accord {agreement:.0%})", votes=votes, errors=errors,
                        usage=usage_summary(meter))}

    def verify_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        signals = case.signal_kinds()
        known = set(state.get("similar_ids") or [])
        diag, relabeled = fix_claims(Diagnosis(**state["diagnosis"]))
        v = verify(diag, state["facts"], case.mapping.features, signals=signals, known_cases=known)
        update: dict[str, Any] = {"verification": v.model_dump()}
        if relabeled:
            update["diagnosis"] = diag.model_dump()
        ok = sum(c.ok for c in v.checks)
        summary = f"{ok}/{len(v.checks)} preuves exactes" + (f" · problèmes : {'; '.join(v.problems)}" if v.problems else "")
        if relabeled:
            summary += f" · libellé corrigé : {', '.join(relabeled)}"
        if not v.ok and self.decider is not None:
            if state.get("revisions", 0) < self.config.max_revisions:
                update.update(revisions=state.get("revisions", 0) + 1,
                              critique="Le vérificateur a trouvé des incohérences : " + "; ".join(v.problems))
            else:
                # Révision épuisée et toujours invérifiable : on n'escalade pas, on écarte le LLM et on
                # applique le décideur à règles, dont chaque preuve est exacte par construction.
                d = diagnose_by_rules(case)
                v2 = verify(d, state["facts"], case.mapping.features, signals=signals, known_cases=known)
                update.update(diagnosis=d.model_dump(), verification=v2.model_dump(), agreement=1.0,
                              decider=f"{state.get('decider', '')} -> règles (preuves du LLM invérifiables)")
                summary += " · repli sur le décideur à règles"
        return {**update, **_step("verify", t0, summary, problems=v.problems)}

    def judge_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        if self.judge is None:
            return _step("judge", t0, "aucun juge configuré")
        case = self._case(state)
        extra = (f"<proposition>\n{compact(state['diagnosis'])}\n</proposition>\n\n"
                 f"<verification_automatique>\n{compact(state.get('verification'))}\n</verification_automatique>")
        meter: list = []
        try:
            v = structured(self.judge, self._messages(prompts.JUDGE, case, state["facts"], similar=state.get("similar"),
                                                      extra=extra), JudgeVerdict, meter=meter)
        except LLMError as exc:
            return _step("judge", t0, f"juge indisponible : {exc}"[:200], usage=usage_summary(meter))
        update: dict[str, Any] = {"verdicts": [v.model_dump()]}
        if v.verdict == "revise" and state.get("revisions", 0) < self.config.max_revisions and self.decider is not None:
            update.update(revisions=state.get("revisions", 0) + 1, critique=f"Le juge demande une révision : {v.critique}")
        return {**update, **_step("judge", t0, f"{v.verdict} · {v.scores} · {v.critique[:160]}",
                                  usage=usage_summary(meter))}

    def policy_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        case = self._case(state)
        verdicts = state.get("verdicts") or []
        verdict = JudgeVerdict(**verdicts[-1]) if verdicts else None
        # Second avis déterministe - seulement s'il est indépendant : si le diagnostic vient déjà des
        # règles, les compter deux fois ne vaudrait pas confirmation.
        from_rules = "règles" in state.get("decider", "") or state.get("decider", "").startswith("rules")
        second = None if from_rules else diagnose_by_rules(case).action
        p = evaluate(Diagnosis(**state["diagnosis"]), case, Verification(**state["verification"]), verdict,
                     state.get("agreement", 0.0), self.config.policy, second_opinion=second)
        failed = [r.rule for r in p.rules if not r.passed]
        if p.mode == "human":
            # On prévient ici et pas dans await_human : LangGraph rejoue ce nœud à la reprise,
            # on enverrait le message deux fois.
            self._notify({**state, "policy": p.model_dump()}, "awaiting_human")
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
        # L'état reçu par le nœud, pas le checkpoint : LangGraph écrit ses checkpoints en asynchrone,
        # relire le checkpoint ici pourrait donner une version en retard d'une étape.
        run = self._assemble(state["run_id"], dict(state))
        action = state["policy"]["final_action"]
        outputs = execute(action, run, case, self.out_dir)
        if action in notify.NOTIFY_ACTIONS:
            outputs["notification"] = self._notify(state, "executed")
        return {"status": "executed", "outputs": outputs,
                **_step("act", t0, f"{action} exécuté" + (f" -> {', '.join(outputs)}" if outputs else ""))}

    def _notify(self, state: dict, event: str) -> str:
        try:
            run = self._assemble(state["run_id"], dict(state))
            return notify.send(run, self._case(state).project, event, self.out_dir)
        except Exception:  # une notification ratée ne doit pas casser la décision
            log.warning("notification impossible", exc_info=True)
            return "échec"

    # -- graphe -------------------------------------------------------------------------------

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        for name, fn in (("prepare", self.prepare), ("investigate", self.investigate), ("diagnose", self.diagnose),
                         ("verify", self.verify_node), ("judge", self.judge_node), ("policy", self.policy_node),
                         ("await_human", self.await_human), ("act", self.act)):
            g.add_node(name, _traced(name, fn))
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
