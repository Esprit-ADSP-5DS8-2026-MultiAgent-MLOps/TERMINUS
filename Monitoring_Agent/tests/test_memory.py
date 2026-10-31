"""Mémoire d'incidents : recherche hybride filtrée, seuil, citations vérifiées, apprentissage des décisions."""

from __future__ import annotations

import pytest
from test_agent import MAPPING, Scripted, approve, frame, make_case, retrain_from

from monitoring_agent.analysis import Analyzer
from monitoring_agent.config import AnalyzerConfig


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer(frame(10000), MAPPING, AnalyzerConfig(window_size=1000))

from monitoring_agent.agent.graph import MonitoringAgent
from monitoring_agent.agent.guards import verify
from monitoring_agent.agent.schemas import Diagnosis, Evidence
from monitoring_agent.memory.cards import IncidentCard, signature
from monitoring_agent.memory.store import HashEmbedder, IncidentMemory, rrf


def _card(cause, action, sig, family="ml", text=""):
    return IncidentCard(id="", source="test", family=family, title=cause, root_cause=cause, action=action,
                        text=text or f"[classification] {cause} {action}", signature=sig)


def test_rrf_rewards_agreement_between_rankings():
    s = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert s["a"] == s["b"] > s["c"] == s["d"]


def test_hybrid_search_filters_family_and_threshold(tmp_path, analyzer):
    mem = IncidentMemory(tmp_path, HashEmbedder())
    case = make_case(analyzer, frame(1000, shift=1.5, flip=0.3))
    from monitoring_agent.agent.context import case_facts
    sig = signature(case_facts(case), case.signal_kinds())
    far = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    mem.add([_card("covariate_shift", "request_retraining", sig),
             _card("data_quality", "notify_data_team", far, text="zzz qqq"),
             _card("service_degradation", "open_incident", sig, family="service")])
    hits = mem.search(case)
    ids = [h["id"] for h in hits]
    assert ids[0] == "INC-0001"                       # même signature : trouvé en premier
    assert "INC-0003" not in ids                      # filtre : un incident de service n'est pas un exemple ML
    assert "INC-0002" not in ids                      # seuil : ni le texte ni les chiffres ne suivent


def test_invented_case_citation_is_caught():
    d = Diagnosis(root_cause="no_issue", summary="calme", evidence=[Evidence(fact="a", value=1.0, claim="c")],
                  action="no_action", rationale="comme INC-0042 et INC-0001", confidence=0.9)
    v = verify(d, {"a": 1.0}, [], known_cases={"INC-0001"})
    assert not v.ok and any("INC-0042" in p for p in v.problems)


def test_agent_retrieves_then_learns_confirmed_decision(tmp_path, analyzer):
    mem = IncidentMemory(tmp_path / "mem", HashEmbedder())
    case = make_case(analyzer, frame(1000, shift=1.5, flip=0.3))
    from monitoring_agent.agent.context import case_facts
    mem.add([_card("covariate_shift", "request_retraining", signature(case_facts(case), case.signal_kinds()))])
    agent = MonitoringAgent(Scripted("d", diagnoses=[retrain_from(case)]), Scripted("j", verdicts=[approve()]),
                            out_dir=tmp_path, memory=mem)
    run = agent.decide(case)
    assert run.status == "executed"
    assert run.trace[0].detail["retrieval"][0]["id"] == "INC-0001"
    assert mem.count() == 2                          # décision confirmée -> nouvelle fiche
    assert mem.all()[-1].source == "agent" and "confirmé" in mem.all()[-1].outcome
