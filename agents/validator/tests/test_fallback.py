"""Bascule automatique Groq -> secours (sans réseau : `post` injecté)."""
import pytest

from agentic_core.llm_client import FallbackClient, GroqClient, LLMError, ScriptedClient, _fallback_specs, build_client


class Resp:
    def __init__(self, code=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = code, body, headers or {}, text

    def json(self):
        return self._body


def ok(model="gemini-2.5-flash"):
    return Resp(body={"choices": [{"message": {"content": "OK"}}], "model": model,
                      "usage": {"prompt_tokens": 10, "completion_tokens": 2}})


def test_quota_429_switches_to_fallback_and_pauses_primary():
    calls = {"groq": 0, "gem": 0}

    def groq_post(*a):
        calls["groq"] += 1
        return Resp(429, headers={"retry-after": "30"}, text="rate limit")      # 30 s > max_retry_wait -> pas d'attente

    def gem_post(*a):
        calls["gem"] += 1
        return ok()

    slept, msgs = [], []
    groq = GroqClient(api_key="g", post=groq_post, sleep=slept.append, max_retry_wait=5)
    gem = GroqClient(api_key="k", model="gemini-2.5-flash", agent="validator", post=gem_post, tpm_limit=0, reasoning_effort=None)
    fb = FallbackClient([groq, gem], on_switch=msgs.append)
    assert fb.chat([{"role": "user", "content": "x"}]).model == "gemini-2.5-flash"
    assert slept == [] and calls == {"groq": 1, "gem": 1} and "bascule" in msgs[0]
    fb.chat([{"role": "user", "content": "x"}])                                  # 2e étape : Groq en pause, on ne le re-sollicite pas
    assert calls == {"groq": 1, "gem": 2}
    assert fb.tpm_limit is None and fb.model == "gemini-2.5-flash"


@pytest.mark.parametrize("code", [413, 400, 401, 500])
def test_other_errors_also_switch(code):
    bad = GroqClient(api_key="g", max_retries=0, post=lambda *a: Resp(code, text="x"))
    fb = FallbackClient([bad, ScriptedClient([{"content": "secours"}], model="backup")])
    assert fb.chat([]).message["content"] == "secours"


def test_all_fail_raises_typed_error_and_never_leaks_key():
    bad = lambda: GroqClient(api_key="SECRET", max_retries=0, post=lambda *a: Resp(429, text="quota"))
    with pytest.raises(LLMError) as e:
        FallbackClient([bad(), bad()]).chat([])
    assert e.value.kind == "quota" and "SECRET" not in str(e.value)


def test_cooldown_expires_and_primary_is_retried():
    t = [0.0]
    prim = ScriptedClient([LLMError("quota", "quota"), {"content": "retour"}], model="groq")
    fb = FallbackClient([prim, ScriptedClient([{"content": "secours"}], model="backup")], cooldown_s=60, clock=lambda: t[0])
    assert fb.chat([]).message["content"] == "secours"
    t[0] = 61
    assert fb.chat([]).message["content"] == "retour"


def test_missing_groq_key_falls_back():
    fb = FallbackClient([GroqClient(api_key=""), ScriptedClient([{"content": "ok"}], model="b")])
    import os
    os.environ.pop("GROQ_API_KEY", None)
    assert fb.chat([]).message["content"] == "ok"


def test_build_client_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("FALLBACK_1_PROVIDER", "gemini")
    monkeypatch.setenv("FALLBACK_1_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("FALLBACK_1_API_KEY", "k")
    c = build_client("validator")
    assert isinstance(c, FallbackClient) and len(c.clients) == 2
    g = c.clients[1]
    assert "generativelanguage" in g.base_url and g.tpm_limit is None and g.max_tokens_key == "max_tokens"
    p = g.build_payload([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert p["model"] == "gemini-2.5-flash" and "reasoning_effort" not in p and p["max_tokens"] == 2500
    assert c.clients[0].max_retry_wait == 5
    monkeypatch.delenv("FALLBACK_1_MODEL")
    assert isinstance(build_client("validator"), GroqClient)                     # sans secours : comportement inchangé


def test_fallback_specs_numbering():
    s = _fallback_specs({"FALLBACK_1_MODEL": "a", "FALLBACK_3_MODEL": "c"})
    assert [x["n"] for x in s] == [1, 3]


def test_dotenv_is_loaded_without_overriding(tmp_path, monkeypatch):
    import importlib, agentic_core
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('# c\nTEST_A=from_file\nexport TEST_B="quoted"\nTEST_C=file\n', encoding="utf-8")
    monkeypatch.setenv("TEST_C", "exported")
    for k in ("TEST_A", "TEST_B"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(agentic_core.Path if hasattr(agentic_core, "Path") else __import__("pathlib").Path, "is_file", lambda self: self == tmp_path / ".env", raising=False)
    agentic_core._load_dotenv()
    import os
    assert os.environ["TEST_A"] == "from_file" and os.environ["TEST_B"] == "quoted" and os.environ["TEST_C"] == "exported"


def test_second_groq_key_keeps_groq_settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g1")
    monkeypatch.setenv("FALLBACK_1_PROVIDER", "groq")
    monkeypatch.setenv("FALLBACK_1_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("FALLBACK_1_API_KEY", "g2")
    c = build_client("validator")
    g2 = c.clients[1]
    assert g2.api_key == "g2" and "groq.com" in g2.base_url and g2.tpm_limit == 8000 and g2.reasoning_effort == "low"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Les tests ne doivent pas dépendre des FALLBACK_* du vrai .env."""
    import os
    for k in list(os.environ):
        if k.startswith(("FALLBACK_", "GROQ_MODEL")):
            monkeypatch.delenv(k, raising=False)


def test_gemini_gets_dummy_signature_and_others_do_not():
    from agentic_core.llm_client import GEMINI_DUMMY_SIGNATURE
    hist = [{"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "f", "arguments": "{}"},
                                                                  "extra_content": {"google": {"thought_signature": "REAL"}}}]}]
    gem = GroqClient(api_key="k", model="gemini-3.8-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                     reasoning_effort=None, tpm_limit=0).build_payload(hist)["messages"]
    assert gem[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == GEMINI_DUMMY_SIGNATURE
    assert gem[2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "REAL"     # vraie signature conservée
    groq = GroqClient(api_key="k").build_payload(hist)["messages"]
    assert "extra_content" not in groq[2]["tool_calls"][0] and "extra_content" in hist[2]["tool_calls"][0]   # original non modifié


def test_llm_order_puts_gemini_first(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    for n, prov, model in ((1, "groq", "openai/gpt-oss-120b"), (2, "gemini", "gemini-3.8-flash")):
        monkeypatch.setenv(f"FALLBACK_{n}_PROVIDER", prov)
        monkeypatch.setenv(f"FALLBACK_{n}_MODEL", model)
        monkeypatch.setenv(f"FALLBACK_{n}_API_KEY", f"k{n}")
    assert [c.model for c in build_client("validator").clients] == ["openai/gpt-oss-120b", "openai/gpt-oss-120b", "gemini-3.8-flash"]
    monkeypatch.setenv("LLM_ORDER", "2,groq,1")
    c = build_client("validator")
    assert [x.model for x in c.clients] == ["gemini-3.8-flash", "openai/gpt-oss-120b", "openai/gpt-oss-120b"]
    assert c.tpm_limit is None and c.clients[1].api_key == "g" and c.clients[2].api_key == "k1"


def test_prefetch_includes_root_python_files(tmp_path):
    from mini_project import make_mini_project
    from policy import load_contract
    from validator_agent.agent import prefetch_facts
    from validator_agent.tools import ValidatorToolBox
    tb = ValidatorToolBox(make_mini_project(tmp_path / "proj"), load_contract("sklearn-tabular"))
    out = prefetch_facts(tb)
    reads = [c["args"] for c in tb.calls if c["tool"] == "read_file"]
    assert any("train.py" in a for a in reads) and all("data.csv" not in a for a in reads) and "def train" in out


def test_framework_evidence_is_regrounded_not_refused(tmp_path):
    from mini_project import REQ_LINE, make_mini_project
    from policy import load_contract
    from validator_agent.agent import Invariants, _make_checker, prefetch_facts
    from validator_agent.tools import ValidatorToolBox
    contract = load_contract("sklearn-tabular")
    tb = ValidatorToolBox(make_mini_project(tmp_path / "proj"), contract)
    prefetch_facts(tb)
    assert tb.is_grounded(REQ_LINE) and tb.is_grounded('"name": "pandas"')     # dépendance brute + JSON préchargés
    check = _make_checker(tb, contract, Invariants(require_all_mandatory_reported=False, cross_check_facts=False))
    checks = [{"check_id": c.id, "status": "PASS"} for c in contract.checks[:1]]
    ok = {"checks": checks, "frameworks": [{"name": "scikit-learn", "evidence": "scikit-learn 1.8.0 (citation approximative)"}]}
    sub, problems = check(ok)
    assert not [p for p in problems if p.startswith("framework")] and sub.frameworks[0].evidence == '"name": "scikit-learn"'
    bad = {"checks": checks, "frameworks": [{"name": "torch", "evidence": "torch==9.9"}]}
    assert any(p.startswith("framework torch") for p in check(bad)[1])                        # une vraie invention reste refusée


def test_groq_schema_accepts_null_but_gemini_schema_is_untouched(tmp_path):
    import json
    from mini_project import make_mini_project
    from policy import load_contract
    from validator_agent.tools import ValidatorToolBox
    tools = ValidatorToolBox(make_mini_project(tmp_path / "proj"), load_contract("sklearn-tabular")).specs()
    groq = GroqClient(api_key="k").build_payload([{"role": "user", "content": "x"}], tools)["tools"]
    run = next(t for t in groq if t["function"]["name"] == "run_in_sandbox")["function"]["parameters"]
    assert run["properties"]["script"]["type"] == ["string", "null"] and run["properties"]["command"]["type"] == "string"   # obligatoire : inchangé
    sub = next(t for t in groq if t["function"]["name"] == "submit_report")["function"]["parameters"]
    item = sub["properties"]["checks"]["items"]["properties"]
    assert item["evidence"]["type"] == ["string", "null"] and item["status"]["type"] == "string" and "severity" in item and None in item["severity"]["enum"]
    gem = GroqClient(api_key="k", model="g", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                     reasoning_effort=None, tpm_limit=0).build_payload([{"role": "user", "content": "x"}], tools)["tools"]
    assert json.dumps(gem) == json.dumps(tools)                                                # Gemini : schéma d'origine
    assert tools[-1]["function"]["parameters"]["properties"]["checks"]["items"]["properties"]["evidence"]["type"] == "string"   # original non modifié


def test_null_arguments_are_dropped():
    from agentic_core.loop import _drop_nulls
    assert _drop_nulls({"command": "harness", "script": None, "modules": [None, "a"], "x": {"y": None, "z": 1}}) == \
        {"command": "harness", "modules": ["a"], "x": {"z": 1}}