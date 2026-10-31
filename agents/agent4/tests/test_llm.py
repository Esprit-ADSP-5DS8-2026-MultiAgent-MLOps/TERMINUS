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


def test_hosted_finetuned_model_needs_its_url_and_gets_facts_only(monkeypatch, tmp_path):
    import pytest as _pytest

    from monitoring_agent.agent import llm
    settings = {"HF_TOKEN": "hf_x"}
    with _pytest.raises(llm.LLMError, match="MONITORING_HF_BASE_URL"):
        llm.build_model("hf:agent4", settings)
    m = llm.build_model("hf:agent4", settings | {"MONITORING_HF_BASE_URL": "https://u-s.hf.space/v1/",
                                                 "MONITORING_HF_TIMEOUT_S": "300"})
    assert m.base_url == "https://u-s.hf.space/v1" and m.timeout_s == 300 and m.api_key == "hf_x"
    assert llm.build_model("hf:agent4", {}) is None                      # sans jeton : pas de modèle
    env = tmp_path / ".env"
    env.write_text("MONITORING_DECIDER=hf:agent4\nMONITORING_HF_BASE_URL=https://u-s.hf.space/v1\n"
                   "HF_TOKEN=hf_x\nGROQ_API_KEY=g\n", encoding="utf-8")
    monkeypatch.setenv("MONITORING_ENV_FILE", str(env))
    decider, _ = llm.decider_and_judge()
    assert decider.local and decider.investigator is not None           # faits seuls, enquête au grand modèle


def test_export_writes_ollama_modelfile_and_space(tmp_path):
    from monitoring_agent.finetune import export
    space = export.prepare_space(tmp_path, "moi/agent4-qwen")
    docker = (space / "Dockerfile").read_text(encoding="utf-8")
    assert "moi/agent4-qwen" in docker and export.GGUF_NAME in docker and '"7860"' in docker
    assert "app_port: 7860" in (space / "README.md").read_text(encoding="utf-8")
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"x")
    stage = export.prepare_repo(tmp_path, adapters, "moi/agent4-qwen", "| modèle | cas |")
    modelfile = (stage / "Modelfile").read_text(encoding="utf-8")
    assert f"FROM ./{export.GGUF_NAME}" in modelfile and "{{ .Role }}" in modelfile
    assert (stage / "adapters" / "adapters.safetensors").exists()
