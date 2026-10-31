"""Tests du client LLM : retries, Retry-After, bascule sur le LLM de secours. Sans reseau."""
import httpx
import pytest

from agent_api_docker import llm
from agent_api_docker.llm import FallbackClient, LLMError, LLMUnavailable, OpenAICompatClient


def _resp(status, json=None, headers=None, text=""):
    return httpx.Response(status, json=json, headers=headers, text=text if json is None else None,
                          request=httpx.Request("POST", "http://x"))


OK = {"model": "m", "choices": [{"message": {"content": "ok", "tool_calls": None}}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    return slept


def script(monkeypatch, per_host):
    """per_host: {'a': [reponses...], 'b': [...]} consommees dans l'ordre ; renvoie le journal des appels."""
    calls = []

    def fake_post(url, **kw):
        host = url.split("//")[1].split("/")[0]
        calls.append(host)
        item = per_host[host].pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(llm.httpx, "post", fake_post)
    return calls


def test_retry_puis_succes(monkeypatch, no_sleep):
    calls = script(monkeypatch, {"a": [_resp(503), _resp(200, OK)]})
    assert OpenAICompatClient("m", "http://a").chat([], []).text == "ok"
    assert calls == ["a", "a"] and len(no_sleep) == 1


def test_retry_after_respecte(monkeypatch, no_sleep):
    script(monkeypatch, {"a": [_resp(429, headers={"retry-after": "3"}), _resp(200, OK)]})
    OpenAICompatClient("m", "http://a").chat([], [])
    assert 3 <= no_sleep[0] <= 3.6


def test_try_again_in_dans_le_corps(monkeypatch, no_sleep):
    body = '{"error":{"message":"Rate limit reached. Please try again in 2.5s."}}'
    script(monkeypatch, {"a": [_resp(429, text=body), _resp(200, OK)]})
    OpenAICompatClient("m", "http://a").chat([], [])
    assert 2.5 <= no_sleep[0] <= 3.1


def test_attente_trop_longue_ne_bloque_pas(monkeypatch, no_sleep):
    calls = script(monkeypatch, {"a": [_resp(429, headers={"retry-after": "45"})]})
    with pytest.raises(LLMUnavailable) as e:
        OpenAICompatClient("m", "http://a").chat([], [])
    assert calls == ["a"] and no_sleep == [] and e.value.retry_after_s == 45


def test_erreur_definitive_non_reessayee(monkeypatch):
    calls = script(monkeypatch, {"a": [_resp(401, text="bad key")]})
    with pytest.raises(LLMError) as e:
        OpenAICompatClient("m", "http://a").chat([], [])
    assert not isinstance(e.value, LLMUnavailable) and calls == ["a"]


def test_bascule_sur_secours_puis_coupe_circuit(monkeypatch):
    calls = script(monkeypatch, {
        "a": [_resp(429, headers={"retry-after": "45"}), ],   # un seul appel autorise vers a
        "b": [_resp(200, {**OK, "model": "secours"}), _resp(200, OK), ],
    })
    msgs = []
    fc = FallbackClient([OpenAICompatClient("principal", "http://a"), OpenAICompatClient("secours", "http://b")],
                        on_switch=msgs.append)
    r1 = fc.chat([], [])
    r2 = fc.chat([], [])            # a est en pause (45 s) : on ne le re-sollicite pas
    assert r1.model == "secours" and r2.text == "ok"
    assert calls == ["a", "b", "b"] and len(msgs) == 1


def test_erreur_definitive_du_principal_ne_bascule_pas(monkeypatch):
    calls = script(monkeypatch, {"a": [_resp(401, text="bad")], "b": [_resp(200, OK)]})
    fc = FallbackClient([OpenAICompatClient("p", "http://a"), OpenAICompatClient("s", "http://b")])
    with pytest.raises(LLMError):
        fc.chat([], [])
    assert calls == ["a"]


def test_tous_en_panne(monkeypatch):
    script(monkeypatch, {"a": [_resp(503)] * 3, "b": [_resp(503)] * 3})
    fc = FallbackClient([OpenAICompatClient("p", "http://a"), OpenAICompatClient("s", "http://b")])
    with pytest.raises(LLMError, match="Tous les LLM"):
        fc.chat([], [])


def test_chaine_de_4_secours(monkeypatch):
    calls = script(monkeypatch, {
        "p": [_resp(503)] * 3, "s1": [_resp(429, headers={"retry-after": "45"})],
        "s2": [_resp(503)] * 3, "s3": [_resp(503)] * 3, "s4": [_resp(200, {**OK, "model": "dernier"})],
    })
    fc = FallbackClient([OpenAICompatClient(n, f"http://{n}") for n in ("p", "s1", "s2", "s3", "s4")])
    assert fc.chat([], []).model == "dernier"
    assert [c for c in calls if c in ("s1", "s4")] == ["s1", "s4"]


def test_make_client_lit_4_secours(monkeypatch):
    for k, v in {"provider": "groq", "model": "m", "api_key": "k"}.items():
        monkeypatch.setattr(llm.config, "LLM_" + k.upper(), v)
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "")
    monkeypatch.setattr(llm.config, "LLM_FALLBACKS", [
        {"n": i, "provider": p, "model": f"m{i}", "base_url": "", "key_env": f"K{i}", "api_key": "x"}
        for i, p in enumerate(["gemini", "openai", "groq", "gemini"], 1)])
    c = llm.make_client()
    assert isinstance(c, FallbackClient) and len(c.clients) == 5
