"""
L'agent avec des LLM SCRIPTÉS (aucun réseau) : on teste le graphe et surtout les garde-fous.
Ce qui compte : même un LLM qui se trompe ou invente ne peut pas faire d'action dangereuse.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from monitoring_agent.agent import prompts
from monitoring_agent.agent.context import Case, call_tool, case_facts
from monitoring_agent.agent.graph import MonitoringAgent, vote
from monitoring_agent.agent.guards import evaluate, verify
from monitoring_agent.agent.llm import LLMResponse, ToolCall
from monitoring_agent.agent.schemas import Diagnosis, Evidence, JudgeVerdict
from monitoring_agent.analysis import Analyzer
from monitoring_agent.config import AnalyzerConfig
from monitoring_agent.schemas import ColumnMapping

RNG = np.random.default_rng(3)
MAPPING = ColumnMapping(task="classification", features=["x", "cat"], target="y", prediction="pred",
                        prediction_proba="proba", primary_metric="roc_auc")


def frame(n, shift=0.0, flip=0.0, new_cat=False):
    x = RNG.normal(shift, 1, n)
    cat = RNG.choice(["a", "b", "c"], n).astype(object)
    if new_cat:
        cat[: n // 5] = "zzz"
    p = 1 / (1 + np.exp(-2 * x))
    y = (RNG.random(n) < p).astype(int)
    if flip:
        idx = RNG.random(n) < flip
        y[idx] = 1 - y[idx]
    return pd.DataFrame({"x": x, "cat": cat, "proba": p, "pred": (p >= 0.5).astype(int), "y": y})


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer(frame(10000), MAPPING, AnalyzerConfig(window_size=1000))


def make_case(analyzer, last: pd.DataFrame, decisions=None) -> Case:
    history = [analyzer.analyze_window(frame(1000), i) for i in range(3)]
    history.append(analyzer.analyze_window(last, 3))
    return Case(project="demo", mapping=MAPPING, history=history, decisions=decisions or [], model_uri="models:/demo/1")


class Scripted:
    """Faux LLM : répond selon le rôle (enquêteur, diagnostiqueur, juge) reconnu au prompt système."""

    def __init__(self, name, diagnoses=None, verdicts=None, investigate=True):
        self.name, self.calls = name, []
        self.diagnoses = list(diagnoses or [])
        self.verdicts = list(verdicts or [])
        self.investigate = investigate

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0):
        system = messages[0]["content"]
        self.calls.append(system[:20])
        if system == prompts.INVESTIGATOR:
            if self.investigate and not any(m.get("role") == "tool" for m in messages):
                return LLMResponse(tool_calls=[ToolCall("c1", "feature_history", {"feature": "x"})],
                                   raw_message={"role": "assistant", "content": "", "tool_calls": [
                                       {"id": "c1", "type": "function",
                                        "function": {"name": "feature_history", "arguments": '{"feature": "x"}'}}]})
            return LLMResponse(text="x a fortement dérivé ; la performance mesurée baisse.")
        if system == prompts.JUDGE:
            v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            return LLMResponse(text=json.dumps(v))
        d = self.diagnoses.pop(0) if len(self.diagnoses) > 1 else self.diagnoses[0]
        return LLMResponse(text=json.dumps(d(messages) if callable(d) else d))


def facts_of(case):
    return case_facts(case)


def approve():
    return {"verdict": "approve", "scores": {"grounding": 5, "causality": 5, "proportionality": 5, "safety": 5},
            "critique": "Solide."}


def retrain_from(case):
    f = facts_of(case)
    return {"root_cause": "covariate_shift", "summary": "x dérive, perf en baisse.", "affected_features": ["x"],
            "evidence": [{"fact": "drift.x.psi", "value": f["drift.x.psi"], "claim": "drift majeur"},
                         {"fact": "perf.realized.roc_auc", "value": f["perf.realized.roc_auc"], "claim": "perf basse"}],
            "action": "request_retraining", "rationale": "Perf sous la limite.", "confidence": 0.85}


# ---------------------------------------------------------------------------
# Garde-fous déterministes
# ---------------------------------------------------------------------------


def test_verifier_catches_invented_numbers(analyzer):
    case = make_case(analyzer, frame(1000, shift=1.5))
    f = facts_of(case)
    good = Diagnosis(root_cause="covariate_shift", summary="", action="open_incident", rationale="", confidence=0.9,
                     evidence=[Evidence(fact="drift.x.psi", value=f["drift.x.psi"], claim="")])
    assert verify(good, f, MAPPING.features).ok
    bad = good.model_copy(update={"evidence": [Evidence(fact="drift.x.psi", value=f["drift.x.psi"] * 3, claim=""),
                                               Evidence(fact="drift.age.psi", value=0.5, claim="")],
                                  "affected_features": ["age"]})
    v = verify(bad, f, MAPPING.features)
    assert not v.ok and len(v.problems) == 3        # mauvais chiffre, fait inexistant, feature inconnue


def test_policy_refuses_retraining_on_drift_alone(analyzer):
    case = make_case(analyzer, frame(1000, shift=0.6))          # drift, mais perf intacte
    assert not case.signal_kinds() & {"performance_drop_realized", "performance_drop_estimated"}
    d = Diagnosis(root_cause="covariate_shift", summary="", action="request_retraining", rationale="",
                  confidence=0.95, evidence=[Evidence(fact="drift.x.psi", value=facts_of(case)["drift.x.psi"], claim="")])
    p = evaluate(d, case, verify(d, facts_of(case), MAPPING.features), JudgeVerdict(verdict="approve"), 1.0)
    assert p.final_action == "increase_monitoring" and p.mode == "auto"


def test_policy_refuses_retraining_on_broken_data(analyzer):
    case = make_case(analyzer, frame(1000, flip=0.4, new_cat=True))
    assert case.has_quality_problem()
    d = Diagnosis(root_cause="concept_drift", summary="", action="request_retraining", rationale="", confidence=0.9,
                  evidence=[Evidence(fact="perf.realized.roc_auc", value=facts_of(case)["perf.realized.roc_auc"], claim="")])
    p = evaluate(d, case, verify(d, facts_of(case), MAPPING.features), JudgeVerdict(verdict="approve"), 1.0)
    assert p.final_action == "notify_data_team"


def test_policy_cooldown_and_rollback(analyzer):
    case = make_case(analyzer, frame(1000, shift=1.5, flip=0.3),
                     decisions=[{"action": "request_retraining", "status": "executed", "mode": "auto",
                                 "created_at": pd.Timestamp.now(tz="UTC").isoformat()}])
    f = facts_of(case)
    d = Diagnosis(**retrain_from(case))
    p = evaluate(d, case, verify(d, f, MAPPING.features), JudgeVerdict(verdict="approve"), 1.0)
    assert p.final_action == "open_incident"                    # déjà une demande en cours
    rb = d.model_copy(update={"action": "rollback_model"})
    assert evaluate(rb, case, verify(rb, f, MAPPING.features), JudgeVerdict(verdict="approve"), 1.0).mode == "human"


def test_vote_measures_agreement():
    a = Diagnosis(root_cause="covariate_shift", summary="", action="request_retraining", rationale="", confidence=0.6)
    b = a.model_copy(update={"confidence": 0.9})
    c = a.model_copy(update={"action": "open_incident"})
    chosen, agreement = vote([a, b, c])
    assert chosen.confidence == 0.9 and agreement == pytest.approx(2 / 3)


def test_tools_are_read_only_and_robust(analyzer):
    case = make_case(analyzer, frame(1000, shift=1.5))
    assert "history" in call_tool(case, "feature_history", {"feature": "x"})
    assert "error" in call_tool(case, "feature_history", {"feature": "inexistante"})
    assert "error" in call_tool(case, "drop_table", {})


# ---------------------------------------------------------------------------
# Graphe complet
# ---------------------------------------------------------------------------


def test_full_graph_executes_retraining_request(analyzer, tmp_path):
    case = make_case(analyzer, frame(1000, shift=1.5, flip=0.3))
    assert "performance_drop_realized" in case.signal_kinds()
    decider = Scripted("fake:decider", diagnoses=[retrain_from(case)])
    judge = Scripted("fake:judge", verdicts=[approve()])
    run = MonitoringAgent(decider, judge, out_dir=tmp_path).decide(case)
    assert run.status == "executed", run.error
    assert run.policy.final_action == "request_retraining" and run.policy.mode == "auto"
    assert [t.node for t in run.trace] == ["prepare", "investigate", "diagnose", "verify", "judge", "policy", "act"]
    req = json.loads((tmp_path / "retraining_request.json").read_text())
    assert req["model_uri"] == "models:/demo/1" and req["reason"] == "performance_drop"
    assert len(run.samples) == 3 and run.agreement == 1.0


def test_hallucination_triggers_one_revision(analyzer, tmp_path):
    case = make_case(analyzer, frame(1000, shift=1.5, flip=0.3))
    wrong = retrain_from(case)
    wrong["evidence"][0]["value"] = 42.0                         # chiffre inventé
    decider = Scripted("fake:decider", diagnoses=[wrong, wrong, wrong, retrain_from(case)])
    run = MonitoringAgent(decider, Scripted("fake:judge", verdicts=[approve()]), out_dir=tmp_path).decide(case)
    nodes = [t.node for t in run.trace]
    assert nodes.count("diagnose") == 2 and nodes.count("verify") == 2
    assert run.verification.ok and run.status == "executed"


def test_judge_rejection_needs_a_human(analyzer, tmp_path):
    case = make_case(analyzer, frame(1000, shift=1.5, flip=0.3))
    reject = {"verdict": "reject", "scores": {"causality": 2}, "critique": "Cause mal établie."}
    agent = MonitoringAgent(Scripted("d", diagnoses=[retrain_from(case)]), Scripted("j", verdicts=[reject]), out_dir=tmp_path)
    run = agent.decide(case)
    assert run.status == "awaiting_human" and not (tmp_path / "retraining_request.json").exists()
    run = agent.resume(run.id, "reject", by="prof")
    assert run.status == "rejected_by_human" and run.human["by"] == "prof"
    assert not (tmp_path / "retraining_request.json").exists()


def test_without_llm_the_rules_decide(analyzer, tmp_path):
    run = MonitoringAgent(None, None, out_dir=tmp_path).decide(make_case(analyzer, frame(1000, shift=0.6)))
    assert run.decider == "rules" and run.policy.final_action == "increase_monitoring" and run.status == "executed"
