"""Noyau commun : schémas v2 (scellé, lignage, export), client LLM (cache, effort, erreurs typées), boucle (échecs = statuts)."""
import json

import pytest
from helpers import ROOT, tc

from agentic_core.export_schemas import export
from agentic_core.llm_client import (AGENT_DEFAULTS, CachedClient, GroqClient, LLMError, ScriptedClient, agent_settings, parse_duration)
from agentic_core.loop import AgentSpec, LoopConfig, run_loop
from agentic_core.progress import ConsoleProgress
from agentic_core.schemas import (AdmissionStatus, FailureKind, JudgeVerdict, MODELS, RepairReport, ValidatorReport, chain_from, load_report,
                                  seal, verify_seal)


# ----------------------------------------------------------------------------- schémas
def test_seal_detects_tampering_and_roundtrips():
    r = seal(ValidatorReport(report_id="a", project="p", status=AdmissionStatus.ACCEPTED, can_proceed=True))
    assert verify_seal(r)
    again = load_report(json.loads(r.model_dump_json()))
    assert verify_seal(again)
    again.status = AdmissionStatus.REJECTED
    assert not verify_seal(again)


def test_lineage_chain_validator_repair_judge_revalidation():
    v1 = seal(ValidatorReport(report_id="v1", project="p", status=AdmissionStatus.REPAIR_REQUIRED))
    rep = seal(RepairReport(repair_id="r1", project="p", status="REPAIRED", lineage=chain_from(v1)))
    jd = seal(JudgeVerdict(verdict_id="j1", repair_id="r1", verdict="ACCEPT", next_action="REVALIDATE", lineage=chain_from(rep)))
    v2 = seal(ValidatorReport(report_id="v2", project="p", status=AdmissionStatus.ACCEPTED, lineage=chain_from(jd)))
    assert [(x.report_type, x.report_id) for x in v2.lineage] == [("validator", "v1"), ("repair", "r1"), ("judge", "j1")]
    assert v2.lineage[1].report_sha256 == rep.report_sha256


def test_judge_verdict_has_no_admission_decision():
    fields = set(JudgeVerdict.model_fields)
    assert "status" not in fields and "can_proceed" not in fields
    assert {"verdict", "criteria", "next_action", "repair_id"} <= fields


def test_exported_json_schemas_are_up_to_date(tmp_path):
    export(tmp_path)
    for name in MODELS:
        assert (tmp_path / f"{name}.schema.json").read_text() == (ROOT / "docs" / "schemas" / f"{name}.schema.json").read_text(), \
            "schémas obsolètes : python -m agentic_core.export_schemas docs/schemas"


# ----------------------------------------------------------------------------- client LLM
class Resp:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = code, body or {}, headers or {}, "x"

    def json(self):
        return self._b


def ok_body(prompt=1000, cached=0):
    return {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "model": "m",
            "usage": {"prompt_tokens": prompt, "completion_tokens": 10, "total_tokens": prompt + 10, "prompt_tokens_details": {"cached_tokens": cached}}}


def test_parse_duration():
    assert parse_duration("7.66s") == pytest.approx(7.66) and parse_duration("1m30.5s") == pytest.approx(90.5) and parse_duration("250ms") == 0.25


def test_per_agent_models_and_env_override(monkeypatch):
    for k in ("GROQ_MODEL", "GROQ_MODEL_VALIDATOR", "GROQ_MODEL_REPAIR", "GROQ_MODEL_JUDGE"):
        monkeypatch.delenv(k, raising=False)
    assert agent_settings("judge")["model"] != agent_settings("repair")["model"]      # modèle différent pour le Judge
    monkeypatch.setenv("GROQ_MODEL_JUDGE", "other/model")
    assert agent_settings("judge")["model"] == "other/model"
    assert agent_settings("judge", "explicit")["model"] == "explicit"


def test_reasoning_effort_and_temperature_in_payload():
    c = GroqClient(api_key="k", model="openai/gpt-oss-120b", reasoning_effort="low", temperature=0.2, post=lambda *a: None)
    p = c.build_payload([{"role": "user", "content": "x"}])
    assert p["reasoning_effort"] == "low" and p["temperature"] == 0.2
    other = GroqClient(api_key="k", model="llama-3.3-70b-versatile", reasoning_effort="low", post=lambda *a: None)
    assert "reasoning_effort" not in other.build_payload([])       # envoyé seulement aux modèles qui le supportent


def test_usage_reports_cached_tokens():
    c = GroqClient(api_key="k", post=lambda *a: Resp(body=ok_body(1000, 800)))
    assert c.chat([{"role": "user", "content": "x"}]).usage["cached_tokens"] == 800


def test_quota_estimate_deducts_cached_tokens():
    sleeps = []
    msgs = [{"role": "user", "content": "x" * 7000}]                  # ~2000 tokens estimés
    c = GroqClient(api_key="k", max_completion_tokens=500, post=lambda *a: Resp(body=ok_body(2000, 1800)), sleep=sleeps.append)
    c.chat(msgs)
    full = c.estimate_quota_tokens(c.build_payload(msgs))
    c2 = GroqClient(api_key="k", max_completion_tokens=500)
    assert full < c2.estimate_quota_tokens(c2.build_payload(msgs)) - 1500   # le préfixe en cache n'est plus compté
    # pacing : assez de quota restant grâce au cache -> aucune attente
    c._remaining_tokens, c._reset_s = 1000.0, 20.0
    c._pace(full)
    assert sleeps == []
    c._remaining_tokens = 100.0
    c._pace(full)
    assert sleeps and sleeps[0] > 20


@pytest.mark.parametrize("code,kind", [(429, "quota"), (401, "auth"), (400, "http")])
def test_http_errors_are_typed_and_key_never_leaks(code, kind):
    c = GroqClient(api_key="SECRET-KEY", max_retries=0, post=lambda *a: Resp(code))
    with pytest.raises(LLMError) as e:
        c.chat([{"role": "user", "content": "x"}])
    assert e.value.kind == kind and "SECRET-KEY" not in str(e.value)


def test_retries_then_succeeds_and_network_error_typed():
    seq = [Resp(429, headers={"retry-after": "1"}), Resp(body=ok_body())]
    slept = []
    assert GroqClient(api_key="k", post=lambda *a: seq.pop(0), sleep=slept.append).chat([]).message["content"] == "hi"
    assert slept == [1.0]

    def boom(*a):
        raise ConnectionError("down")
    with pytest.raises(LLMError) as e:
        GroqClient(api_key="k", max_retries=0, post=boom).chat([])
    assert e.value.kind == "network"


def test_cached_client_records_and_replays(tmp_path):
    inner = ScriptedClient([{"content": "a"}])
    c = CachedClient(inner, tmp_path)
    m = [{"role": "user", "content": "q"}]
    assert c.chat(m).message["content"] == "a" and not c.chat(m).cached is False
    replay = CachedClient(ScriptedClient([]), tmp_path, mode="replay")
    assert replay.chat(m).cached
    with pytest.raises(LLMError):
        replay.chat([{"role": "user", "content": "inconnu"}])


# ----------------------------------------------------------------------------- boucle générique : échecs = statuts
class Box:
    def specs(self):
        return []

    def call(self, name, args):
        return json.dumps({"ok": True})


def spec(check=lambda a: (a, [])):
    return AgentSpec(name="validator", system_prompt="s", user_message="u", toolbox=Box(), submit_tool="submit", check_submission=check)


@pytest.mark.parametrize("err,kind", [(LLMError("q", "quota"), FailureKind.QUOTA), (LLMError("t", "timeout"), FailureKind.TIMEOUT),
                                      (LLMError("n", "network"), FailureKind.LLM_ERROR), (RuntimeError("bug"), FailureKind.LLM_ERROR)])
def test_loop_turns_errors_into_statuses_not_exceptions(err, kind):
    r = run_loop(ScriptedClient([err]), spec(), LoopConfig())
    assert r.failure and r.failure.kind == kind and r.submission is None


def test_loop_budget_steps_and_no_tool_use():
    big = ScriptedClient([tc("x", {}), tc("x", {}, 2)], tokens_per_call=100)
    assert run_loop(big, spec(), LoopConfig(token_budget=150)).failure.kind == FailureKind.BUDGET_EXCEEDED
    assert run_loop(ScriptedClient([tc("x", {}, i) for i in range(5)]), spec(), LoopConfig(max_steps=3)).failure.kind == FailureKind.MAX_STEPS
    assert run_loop(ScriptedClient([{"content": "bla"}] * 4), spec(), LoopConfig()).failure.kind == FailureKind.NO_TOOL_USE


def test_loop_rejects_then_accepts_submission_and_counts_cached_tokens():
    seen = []

    def check(a):
        seen.append(a)
        return (a, []) if a.get("ok") else (a, ["manque ok"])
    c = ScriptedClient([tc("submit", {"ok": False}), tc("submit", {"ok": True}, 2)], cached_per_call=50)
    r = run_loop(c, spec(check), LoopConfig())
    assert r.failure is None and r.submission == {"ok": True} and r.stats["submit_retries"] == 1 and r.stats["cached_tokens"] == 100


def test_progress_shows_agent_and_cached_tokens(capsys):
    import sys
    run_loop(ScriptedClient([tc("submit", {})], cached_per_call=40), spec(), LoopConfig(), ConsoleProgress(sys.stderr))
    err = capsys.readouterr().err
    assert "[VALIDATOR]" in err and "dont 40 en cache" in err


def test_defaults_declare_all_three_agents():
    assert set(AGENT_DEFAULTS) == {"validator", "repair", "judge"}
