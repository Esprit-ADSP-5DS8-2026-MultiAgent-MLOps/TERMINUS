"""Limite de tokens Groq (TPM) : plafond dur, calibrage, 413, compactage du contexte. Autonome (n'utilise aucun helper du projet)."""
import json

import pytest

from agentic_core.llm_client import GroqClient, LLMError, ScriptedClient
from agentic_core.loop import AgentSpec, LoopConfig, compact_messages, run_loop
from agentic_core.schemas import FailureKind


class Resp:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = code, body or {}, headers or {}, "x"

    def json(self):
        return self._b


def ok_body(prompt=1000):
    return {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "model": "m",
            "usage": {"prompt_tokens": prompt, "completion_tokens": 10, "total_tokens": prompt + 10, "prompt_tokens_details": {"cached_tokens": 0}}}


def big(n):
    return [{"role": "user", "content": "x" * n}]


def tc(name, args, i=1):
    return {"content": "", "tool_calls": [{"id": f"c{i}", "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_guard_shrinks_completion_to_fit_the_tpm_limit():
    sent = []

    def post(url, headers, payload, timeout):
        sent.append(payload["max_completion_tokens"])
        return Resp(body=ok_body(prompt=4000))
    c = GroqClient(api_key="k", tpm_limit=8000, max_completion_tokens=2500, post=post)
    c.chat(big(12_000))
    c.chat(big(17_000))
    assert sent[0] == 2500 and sent[1] < 2500 and sent[1] >= GroqClient.MIN_COMPLETION


def test_request_that_cannot_fit_raises_too_large_before_calling_the_api():
    called = []
    c = GroqClient(api_key="k", tpm_limit=8000, post=lambda *a: called.append(1) or Resp(body=ok_body()))
    with pytest.raises(LLMError) as e:
        c.chat(big(24_000))
    assert e.value.kind == "too_large" and called == []


def test_413_from_groq_is_typed_and_teaches_the_real_limit():
    r = Resp(413)
    r.text = '{"error":{"message":"Request too large ... (TPM): Limit 8000, Requested 10429"}}'
    c = GroqClient(api_key="k", tpm_limit=None, max_retries=0, post=lambda *a: r)
    with pytest.raises(LLMError) as e:
        c.chat(big(100))
    assert e.value.kind == "too_large" and c.tpm_limit == 8000


def test_tpm_limit_from_env_and_disable(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "12000")
    assert GroqClient(api_key="k").tpm_limit == 12000
    monkeypatch.setenv("GROQ_TPM_LIMIT", "0")
    assert GroqClient(api_key="k").tpm_limit is None
    monkeypatch.delenv("GROQ_TPM_LIMIT")
    assert GroqClient(api_key="k").tpm_limit == 8000


def test_estimator_calibrates_on_real_usage():
    c = GroqClient(api_key="k", tpm_limit=8000, post=lambda *a: Resp(body=ok_body(prompt=1000)))
    before = c.prompt_estimate(big(4000))
    c.chat(big(4000))
    assert c.prompt_estimate(big(4000)) < before


def convo(n_tools=5, size=2000):
    m = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(n_tools):
        m.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]})
        m.append({"role": "tool", "tool_call_id": f"c{i}", "content": "y" * size})
    return m


def test_compaction_archives_old_outputs_and_keeps_recent_ones():
    m = convo()
    assert compact_messages(m, keep_last=2) == 3
    tools = [x for x in m if x["role"] == "tool"]
    assert all(t["content"].startswith("[archivé] sortie de read_file") for t in tools[:3]) and all(t["content"] == "y" * 2000 for t in tools[3:])
    assert m[0]["content"] == "sys" and m[1]["content"] == "go" and compact_messages(m, keep_last=2) == 0


class TpmClient(ScriptedClient):
    def __init__(self, script, limit=8000):
        super().__init__(script)
        self.tpm_limit, self.sizes = limit, []

    def prompt_estimate(self, messages, tools=None):
        return len(json.dumps(messages)) // 3

    def chat(self, messages, tools=None):
        est = self.prompt_estimate(messages, tools)
        self.sizes.append(est)
        if est > self.tpm_limit - 700:
            raise LLMError("trop grand", "too_large")
        return super().chat(messages, tools)


class BigOutputBox:
    def specs(self):
        return []

    def call(self, name, args):
        return "o" * 6000


def _spec(box):
    return AgentSpec(name="validator", system_prompt="s", user_message="u", toolbox=box, submit_tool="submit", check_submission=lambda a: (a, []))


def test_loop_survives_a_growing_context_under_a_tpm_limit():
    seq = [tc("read_file", {"path": f"f{i}"}, i) for i in range(8)] + [tc("submit", {"ok": True}, 99)]
    client = TpmClient(seq)
    r = run_loop(client, _spec(BigOutputBox()), LoopConfig(max_steps=12, token_budget=10**9))
    assert r.failure is None and r.submission == {"ok": True} and max(client.sizes) <= 8000 - 700


def test_loop_reports_a_clear_status_when_even_compaction_cannot_help():
    class Always(ScriptedClient):
        tpm_limit = 8000

        def chat(self, messages, tools=None):
            raise LLMError("trop grand", "too_large")
    r = run_loop(Always([]), _spec(BigOutputBox()), LoopConfig())
    assert r.failure.kind == FailureKind.QUOTA and "GROQ_TPM_LIMIT" in r.failure.message
