"""Client LLM sans réseau : sortie contrainte par JSON Schema, replis, et format des prompts v2."""

from __future__ import annotations

import json

import httpx

from monitoring_agent.agent import llm, prompts
from monitoring_agent.agent.llm import OpenAICompatModel, llm_schema, structured
from monitoring_agent.agent.schemas import Diagnosis, JudgeVerdict

GOOD = {"root_cause": "no_issue", "summary": "rien", "affected_features": [], "evidence": [],
        "action": "no_action", "rationale": "calme", "confidence": 0.9}


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def test_schema_is_closed_and_portable():
    s = llm_schema(Diagnosis)
    assert "$defs" not in json.dumps(s) and "$ref" not in json.dumps(s)
    assert set(s["required"]) == set(s["properties"])                 # tous les champs attendus
    assert "request_retraining" in s["properties"]["action"]["enum"]  # vocabulaire fermé imposé au décodage
    j = llm_schema(JudgeVerdict)
    assert j["properties"]["scores"]["required"] == ["grounding", "causality", "proportionality", "safety"]


def test_json_schema_sent_then_fallback_to_json_object(monkeypatch):
    bodies = []

    def post(url, json, headers, timeout):
        bodies.append(json)
        if json["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text='{"error": {"message": "response_format json_schema not supported"}}')
        return _ok(__import__("json").dumps(GOOD))

    monkeypatch.setattr(llm.httpx, "post", post)
    m = OpenAICompatModel("ollama", "qwen3-4b", None)
    d = structured(m, [{"role": "user", "content": "x"}], Diagnosis)
    assert d.action == "no_action"
    assert [b["response_format"]["type"] for b in bodies] == ["json_schema", "json_object"]
    assert m.schema_ok is False                                        # mémorisé : plus de json_schema ensuite


def test_groq_rejected_generation_is_repaired(monkeypatch):
    calls = []

    def post(url, json, headers, timeout):
        calls.append(json)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"code": "json_validate_failed",
                                                       "failed_generation": '{"root_cause": "inventé"}'}})
        return _ok(__import__("json").dumps(GOOD))

    monkeypatch.setattr(llm.httpx, "post", post)
    d = structured(OpenAICompatModel("groq", "openai/gpt-oss-120b", "k"), [{"role": "user", "content": "x"}], Diagnosis)
    assert d.root_cause == "no_issue"
    assert "invalide" in calls[1]["messages"][-1]["content"]           # la 2e requête contient la critique


def test_user_message_tags_isolate_data():
    msg = prompts.user_message("résumé", "a = 1", notes="note", similar="INC-1", critique="corrige")
    for tag in ("dossier", "faits", "note_enquete", "cas_similaires", "critique"):
        assert f"<{tag}>" in msg and f"</{tag}>" in msg
    assert msg.index("<critique>") > msg.index("<faits>")              # la critique arrive en dernier
    assert "<note_enquete>" not in prompts.user_message("r", "f")      # pas de balise vide
