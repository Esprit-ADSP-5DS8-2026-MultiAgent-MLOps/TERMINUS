"""
Tests du superviseur : le décideur (règles, Gemini...) choisit la suite,
mais il ne peut jamais sortir des actions autorisées.

Les décideurs « LLM » sont simulés ici : aucun appel réseau, aucune clé nécessaire.
"""

from pathlib import Path

import pytest

from orchestrator import llm
from orchestrator.graph import Run, run_pipeline
from orchestrator.llm import GeminiDecider, LLMDecision

ROOT = Path(__file__).resolve().parents[2]
CREDIT_RISK = str(ROOT / "eval" / "projects" / "credit_risk")


def approve(question):
    return {"decision": "approve", "by": "pytest", "comment": ""}


def reject(question):
    return {"decision": "reject", "by": "pytest", "comment": ""}


def use_decider(monkeypatch, choose):
    """Remplace le décideur « gemini » par un faux LLM : choose(options, rule) -> action."""

    class FakeLLM:
        name = "gemini"

        def decide(self, context, options, rule_choice):
            return LLMDecision(action=choose(options, rule_choice), explanation="explication du faux LLM")

    monkeypatch.setitem(llm.DECIDERS, "gemini", FakeLLM)


def steps(summary):
    return [h["node"] for h in summary["history"]]


# --- mode par défaut ----------------------------------------------------------------


def test_default_mode_is_gemini_when_a_key_is_available(monkeypatch):
    monkeypatch.delenv("ORCH_SUPERVISOR", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-test")
    assert llm.default_mode() == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert llm.default_mode() == "rules"  # sans clé, le pipeline tourne quand même
    monkeypatch.setenv("ORCH_SUPERVISOR", "rules")
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-test")
    assert llm.default_mode() == "rules"  # le choix explicite gagne


# --- mode règles ---------------------------------------------------------------


def test_rules_mode_records_one_decision_per_step(tmp_path):
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), simulate={})
    assert s["final_status"] == "DONE"
    assert s["supervisor_mode"] == "rules"
    assert [d["action"] for d in s["supervision"]] == ["agent1", "agent2", "gate_champion", "agent3", "agent4", "end_done"]
    assert all(d["decider"] == "rules" and d["fallback"] is None for d in s["supervision"])


def test_rules_mode_stops_on_a_transient_failure(tmp_path):
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), simulate={"failure": "transient"})
    assert s["final_status"] == "FAILED"  # les règles ne savent pas qu'une erreur réseau est passagère


# --- mode LLM : ce qu'il peut faire ----------------------------------------------------


def test_llm_retries_a_transient_failure(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "retry" if "retry" in opts else rule)
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), supervisor="gemini",
                     simulate={"failure": "transient"})
    assert s["final_status"] == "DONE"
    assert steps(s).count("agent3") == 2
    assert s["attempts"] == {"agent3": 1}


def test_llm_can_hand_a_failure_to_a_human(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "gate_failure" if "gate_failure" in opts else rule)
    run = Run.start(CREDIT_RISK, runs_dir=str(tmp_path), supervisor="gemini", simulate={"failure": "transient"})
    run.answer(approve(None))  # mise en production
    assert run.question["gate"] == "failure_review"
    assert run.question["failed_step"] == "agent3"
    assert "Docker Hub" in run.question["reasons"][0]
    run.answer(approve(None))  # l'humain fait relancer
    assert run.status == "DONE"


def test_human_can_stop_after_a_failure(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "gate_failure" if "gate_failure" in opts else rule)
    s = run_pipeline(CREDIT_RISK, lambda q: reject(q) if q["gate"] == "failure_review" else approve(q),
                     runs_dir=str(tmp_path), supervisor="gemini", simulate={"failure": "permanent"})
    assert s["final_status"] == "FAILED"


def test_llm_may_skip_a_retraining_request(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "end_done" if "end_done" in opts else rule)
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), supervisor="gemini", simulate={"drift": True})
    assert s["final_status"] == "DONE"
    assert [d["gate"] for d in s["decisions"]] == ["champion_replacement"]


# --- mode LLM : ce qu'il ne peut PAS faire ----------------------------------------------


def test_llm_cannot_skip_the_human_validation(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "agent3")  # essaie de déployer sans validation
    run = Run.start(CREDIT_RISK, runs_dir=str(tmp_path), supervisor="gemini", simulate={})
    assert run.question["gate"] == "champion_replacement"  # la pause a bien lieu
    refused = [d for d in run.state["supervision"] if d["fallback"]]
    assert refused and "non autorisée" in refused[0]["fallback"]


def test_llm_cannot_deploy_a_rejected_model(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "agent3")
    s = run_pipeline(CREDIT_RISK, reject, runs_dir=str(tmp_path), supervisor="gemini", simulate={})
    assert s["final_status"] == "REJECTED"
    assert "agent3" not in steps(s)


def test_retries_are_limited(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: "retry")  # réessaierait à l'infini
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), supervisor="gemini",
                     simulate={"failure": "permanent"})
    assert s["final_status"] == "FAILED"
    assert steps(s).count("agent3") == 3  # 1 essai + 2 nouvelles tentatives


def test_a_broken_llm_falls_back_to_the_rules(tmp_path, monkeypatch):
    def broken(opts, rule):
        raise TimeoutError("API injoignable")

    use_decider(monkeypatch, broken)
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), supervisor="gemini", simulate={})
    assert s["final_status"] == "DONE"
    assert all("TimeoutError" in d["fallback"] for d in s["supervision"])


def test_the_mode_is_kept_when_a_run_is_resumed(tmp_path, monkeypatch):
    use_decider(monkeypatch, lambda opts, rule: rule)
    run = Run.start(CREDIT_RISK, runs_dir=str(tmp_path), supervisor="gemini", simulate={})
    reloaded = Run.load(run.run_id, runs_dir=str(tmp_path))
    reloaded.answer(approve(None))
    assert reloaded.status == "DONE"
    assert {d["decider"] for d in reloaded.state["supervision"]} == {"gemini"}


def test_unknown_mode_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="inconnu"):
        Run.start(CREDIT_RISK, runs_dir=str(tmp_path), supervisor="chatgpt")


# --- enregistrement de ce que le décideur a vu -----------------------------------------------


def test_each_decision_records_what_the_decider_saw(tmp_path, monkeypatch):
    seen = []

    class SpyLLM:
        name = "gemini"

        def decide(self, context, options, rule_choice):
            seen.append((context, dict(options)))
            return LLMDecision(action="retry" if "retry" in options else rule_choice, explanation="ok")

    monkeypatch.setitem(llm.DECIDERS, "gemini", SpyLLM)
    s = run_pipeline(CREDIT_RISK, approve, runs_dir=str(tmp_path), supervisor="gemini",
                     simulate={"failure": "transient"})
    assert [(d["context"], d["options"]) for d in s["supervision"]] == seen
    failed = next(d for d in s["supervision"] if d["action"] == "retry")
    assert failed["context"]["resultat"]["statut"] == "FAILED"
    assert "Docker Hub" in failed["context"]["resultat"]["raisons"][0]
    assert set(failed["options"]) == {"end_failed", "retry", "gate_failure"}


def test_export_decisions_writes_one_line_per_decision(tmp_path, monkeypatch, capsys):
    import json

    from orchestrator.__main__ import main

    use_decider(monkeypatch, lambda opts, rule: rule)
    runs = str(tmp_path / "runs")
    run_pipeline(CREDIT_RISK, approve, runs_dir=runs, supervisor="gemini", simulate={})
    run_pipeline(CREDIT_RISK, approve, runs_dir=runs, simulate={})  # mode règles
    out = tmp_path / "decisions.jsonl"

    assert main(["--runs-dir", runs, "export-decisions", "-o", str(out), "--supervisor", "gemini"]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and {r["decider"] for r in rows} == {"gemini"}
    assert {"run_id", "context", "options", "action", "explanation", "rule_choice"} <= set(rows[0])

    assert main(["--runs-dir", runs, "export-decisions", "-o", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 12


# --- GeminiDecider, sans réseau ---------------------------------------------------------


class FakeResponse:
    def __init__(self, text):
        self.text, self.parsed = text, None


def fake_gemini(text):
    decider = GeminiDecider(api_key="test")
    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse(text)

    decider._client = type("Client", (), {"models": Models()})()
    return decider, calls


def test_gemini_answer_is_parsed():
    decider, calls = fake_gemini('{"action": "retry", "explanation": "Erreur réseau passagère."}')
    d = decider.decide({"etape_terminee": "Agent 3"}, {"retry": "Relancer", "end_failed": "Arrêter"}, "end_failed")
    assert d.action == "retry"
    config = calls[0]["config"]
    assert config.temperature == 0 and config.response_mime_type == "application/json"
    assert "retry : Relancer" in calls[0]["contents"]


def test_gemini_garbage_raises_so_the_rules_take_over():
    decider, _ = fake_gemini("je ne sais pas")
    with pytest.raises(Exception):
        decider.decide({}, {"end_failed": "Arrêter"}, "end_failed")


def test_gemini_without_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiDecider().decide({}, {"end_failed": "Arrêter"}, "end_failed")


def test_gemini_html_entities_are_decoded():
    decider, _ = fake_gemini('{"action": "agent1", "explanation": "Le mod&egrave;le n&eacute;cessite l&#39;avis"}')
    d = decider.decide({}, {"agent1": "Lancer l'Agent 1"}, "agent1")
    assert d.explanation == "Le modèle nécessite l'avis"
