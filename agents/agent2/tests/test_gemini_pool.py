import pytest

from mlflow_agent.config import resolve_llm
from mlflow_agent.llm_client import GeminiClient, GeminiPool, LLMError, QuotaError, make_client


class Stub:
    def __init__(self, model, outcome):
        self._model, self.outcome, self.calls = model, outcome, 0

    def complete(self, system, user):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET123")
    monkeypatch.delenv("MLFLOW_AGENT_LLM", raising=False)


def pool(*stubs):
    p = GeminiPool([s._model for s in stubs])
    p._clients = list(stubs)
    return p


def test_next_model_used_after_quota():
    a, b = Stub("a", QuotaError("quota")), Stub("b", "B")
    p = pool(a, b)
    assert p.complete("s", "u") == "B" and p.last_model == "b"


def test_quota_model_is_skipped_on_next_call():
    a, b = Stub("a", QuotaError("quota")), Stub("b", "B")
    p = pool(a, b)
    p.complete("s", "u")
    p.complete("s", "u")
    assert a.calls == 1 and b.calls == 2


def test_all_models_out_of_quota_raises():
    p = pool(Stub("a", QuotaError("q")), Stub("b", QuotaError("q")))
    with pytest.raises(LLMError):
        p.complete("s", "u")


def test_missing_model_is_skipped_for_good():
    a, b = Stub("a", LLMError("gemini HTTP 404: gone")), Stub("b", "B")
    p = pool(a, b)
    p.complete("s", "u")
    p.complete("s", "u")
    assert a.calls == 1


def test_other_error_moves_on_but_is_retried_next_call():
    a, b = Stub("a", LLMError("gemini call failed: URLError")), Stub("b", "B")
    p = pool(a, b)
    p.complete("s", "u")
    p.complete("s", "u")
    assert a.calls == 2


def test_make_client_routing():
    assert isinstance(make_client("gemini:a"), GeminiClient)
    assert isinstance(make_client("gemini:a,b"), GeminiPool)
    assert make_client("gemini:a,b").name == "gemini:a,b"


def test_config_accepts_a_model_list(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('llm = "gemini:a,b,c"')
    assert resolve_llm(None, f) == ("gemini:a,b,c", "config file")
