#!/usr/bin/env python3
"""Applique le correctif « limite de tokens Groq (TPM) » sur VOTRE version du projet. Idempotent, avec sauvegarde.

    python3 apply_tpm_fix.py            # applique (sauvegarde dans ../<projet>_backup_tpm.tar.gz)
    python3 apply_tpm_fix.py --check    # vérifie seulement (ne modifie rien)

Modifie : agentic_core/llm_client.py, agentic_core/loop.py, agentic_core/toolkit.py, repair_agent/agent.py ; crée tests/test_tpm_limit.py.
Toutes les modifications sont faites en mémoire ; si UNE ancre est introuvable, RIEN n'est écrit.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(".")
CHECK = "--check" in sys.argv
if not (ROOT / "pyproject.toml").exists() or not (ROOT / "agentic_core").is_dir():
    sys.exit("ERREUR : lancez ce script depuis la racine du projet (pyproject.toml + agentic_core/).")

edits: dict[pathlib.Path, str] = {}
report: list[str] = []
errors: list[str] = []


def load(rel):
    p = ROOT / rel
    if not p.exists():
        errors.append(f"{rel} : fichier introuvable")
        return None, None
    return p, edits.get(p, p.read_text(encoding="utf-8"))


def patch(rel, marker, old, new, label, count=1):
    """Remplace `old` par `new`. `marker` : texte dont la présence prouve que le patch est déjà appliqué."""
    p, text = load(rel)
    if p is None:
        return
    if marker in text:
        report.append(f"  déjà appliqué : {label}")
        return
    if text.count(old) != count:
        errors.append(f"{rel} : ancre introuvable pour « {label} » (trouvée {text.count(old)}×, attendue {count}×) — votre fichier a divergé")
        return
    edits[p] = text.replace(old, new, count)
    report.append(f"  appliqué      : {label}")


# =========================================================================================== agentic_core/llm_client.py
F = "agentic_core/llm_client.py"
patch(F, "DEFAULT_TPM_LIMIT", '''    CHARS_PER_TOKEN = 3.5
''', '''    CHARS_PER_TOKEN = 3.5          # estimation pour la cadence (attente)
    GUARD_CHARS_PER_TOKEN = 3.0    # estimation PRUDENTE pour le plafond dur (on préfère surestimer)
    MIN_COMPLETION = 700           # en dessous, une réponse (rapport complet) ne tient plus
    SAFETY_MARGIN = 150
    DEFAULT_TPM_LIMIT = 8000       # palier gratuit « on_demand » ; GROQ_TPM_LIMIT=0 désactive ; à régler selon votre palier/modèle
''', "constantes TPM")
patch(F, 'tpm_limit="__default__"', "post=None, sleep=time.sleep):", 'post=None, sleep=time.sleep, tpm_limit="__default__"):', "paramètre tpm_limit du constructeur")
patch(F, "self.tpm_limit = tpm_limit or None", "        self.timeout, self.max_retries = timeout, max_retries\n", '''        self.timeout, self.max_retries = timeout, max_retries
        if tpm_limit == "__default__":
            env = os.environ.get("GROQ_TPM_LIMIT")
            tpm_limit = int(env) if env and env.isdigit() else self.DEFAULT_TPM_LIMIT
        self.tpm_limit = tpm_limit or None             # tokens/minute autorisés pour CE modèle (requête unique incluse)
''', "lecture de GROQ_TPM_LIMIT")
patch(F, "self._ratio = None", "        self._last_cached_tokens = 0\n", "        self._last_cached_tokens = 0\n        self._ratio = None                              # caractères par token, mesuré\n", "état du ratio mesuré")
patch(F, "def _guard_request_size", "    def estimate_quota_tokens(self, payload) -> float:", '''    def prompt_estimate(self, messages, tools=None) -> int:
        """Estimation PRUDENTE (par excès) des tokens d'entrée d'une requête."""
        chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))
        # avant le 1er appel : ratio prudent ; ensuite : ratio MESURÉ sur les `usage` réels de Groq, avec 8 % de marge
        ratio = self._ratio * 0.92 if self._ratio else self.GUARD_CHARS_PER_TOKEN
        return int(chars / ratio)

    def _guard_request_size(self, payload):
        """Plafond dur : prompt + réponse maximale doit tenir dans la limite TPM, sinon Groq répond 413 (attendre n'y change rien).
        On réduit donc `max_completion_tokens` ; si même le minimum ne tient pas, on lève `too_large` AVANT l'appel (la boucle compacte le contexte)."""
        if not self.tpm_limit:
            return
        prompt = self.prompt_estimate(payload["messages"], payload.get("tools"))
        room = self.tpm_limit - prompt - self.SAFETY_MARGIN
        if room < self.MIN_COMPLETION:
            raise LLMError(f"requête estimée à ~{prompt} tokens d'entrée pour une limite de {self.tpm_limit} tokens/minute : "
                           "contexte à réduire", "too_large")
        payload["max_completion_tokens"] = int(min(self.max_completion_tokens, room))

    def estimate_quota_tokens(self, payload) -> float:''', "estimation + plafond dur")
patch(F, "self._guard_request_size(payload)", '''        payload = self.build_payload(messages, tools)
        headers =''', '''        payload = self.build_payload(messages, tools)
        self._guard_request_size(payload)
        headers =''', "appel du plafond avant l'envoi")
patch(F, "self._ratio = max(2.0", '''                self._last_prompt_tokens, self._last_cached_tokens = usage["prompt_tokens"], cached''', '''                self._last_prompt_tokens, self._last_cached_tokens = usage["prompt_tokens"], cached
                if usage["prompt_tokens"] > 200:
                    sent = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))
                    self._ratio = max(2.0, min(sent / usage["prompt_tokens"], 6.0))''', "calibrage sur les tokens réels")
patch(F, 'if code == 413:', '''            body = (getattr(resp, "text", "") or "")[:300]
''', '''            body = (getattr(resp, "text", "") or "")[:300]
            if code == 413:
                m = re.search(r"Limit (\\d+), Requested (\\d+)", body)
                if m:
                    self.tpm_limit = int(m.group(1))       # la vraie limite du palier : les prochains appels s'y adaptent
                raise LLMError(f"HTTP 413 : requête trop grande pour la limite TPM. {body}", "too_large")
''', "gestion du 413")
patch(F, "def prompt_estimate(self, messages, tools=None):\n        f = getattr", '''        self.model = getattr(inner, "model", "cached")
        self.hits = self.misses = 0
''', '''        self.model = getattr(inner, "model", "cached")
        self.hits = self.misses = 0

    @property
    def tpm_limit(self):
        return getattr(self.inner, "tpm_limit", None)

    def prompt_estimate(self, messages, tools=None):
        f = getattr(self.inner, "prompt_estimate", None)
        return f(messages, tools) if f else 0
''', "transmission par CachedClient")
for agent, old_effort in (("validator", '"validator": {"model": "openai/gpt-oss-120b", "reasoning_effort": "medium"'), ("repair", '"repair": {"model": "openai/gpt-oss-120b", "reasoning_effort": "medium"')):
    patch(F, old_effort.replace("medium", "low"), old_effort, old_effort.replace("medium", "low"), f"reasoning_effort=low ({agent})")
p, t = load(F)
if p and "import re" not in (edits.get(p, t)).split("class LLMError")[0]:
    edits[p] = edits.get(p, t).replace("import os\n", "import os\nimport re\n", 1)
    report.append("  appliqué      : import re")

# =========================================================================================== agentic_core/loop.py
F = "agentic_core/loop.py"
patch(F, "def compact_messages", "def _fail(res: LoopResult", '''COMPACT_MIN_CHARS = 300
COMPLETION_RESERVE = 2300        # tokens à garder libres pour la réponse (rapport final + raisonnement)


def compact_messages(messages: list[dict], keep_last: int = 2) -> int:
    """Archive les anciennes sorties d'outils (et les très gros arguments d'appels refusés) pour alléger le contexte.
    Les `keep_last` sorties les plus récentes restent intactes. Retourne le nombre de messages réduits.
    Le LLM peut rappeler l'outil s'il a besoin d'un extrait exact pour une preuve (les preuves sont vérifiées contre TOUTES les sorties passées)."""
    names = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            names[c.get("id")] = (c.get("function") or {}).get("name", "outil")
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    done = 0
    for i in tool_idx[:max(0, len(tool_idx) - keep_last)]:
        c = messages[i].get("content") or ""
        if len(c) > COMPACT_MIN_CHARS and not c.startswith("[archivé"):
            messages[i] = {**messages[i], "content": f"[archivé] sortie de {names.get(messages[i].get('tool_call_id'), 'outil')} ({len(c)} car.) "
                                                      "retirée pour tenir dans la limite de tokens : rappelle l'outil si tu en as besoin (ex. pour citer une preuve exacte)."}
            done += 1
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and i != last_assistant:
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                if len(fn.get("arguments") or "") > 600 and "archivé" not in fn["arguments"]:
                    fn["arguments"] = json.dumps({"archivé": True, "note": "arguments volumineux retirés du contexte"})
                    done += 1
    return done


def _make_room(client, messages, tools) -> None:
    """Avant l'appel : si le contexte estimé laisse moins que COMPLETION_RESERVE tokens pour la réponse, on compacte progressivement."""
    limit, est = getattr(client, "tpm_limit", None), getattr(client, "prompt_estimate", None)
    if not limit or not est:
        return
    for keep in (3, 2, 1, 0):
        if est(messages, tools) <= limit - COMPLETION_RESERVE or not compact_messages(messages, keep):
            if est(messages, tools) <= limit - COMPLETION_RESERVE:
                return


def _fail(res: LoopResult''', "compactage du contexte")
patch(F, 'if e.kind != "too_large":', '''                resp = client.chat(messages, spec.toolbox.specs())
''', '''                tools = spec.toolbox.specs()
                _make_room(client, messages, tools)
                try:
                    resp = client.chat(messages, tools)
                except LLMError as e:
                    if e.kind != "too_large":
                        raise
                    for keep in (1, 0):                                  # requête trop grande : on archive tout l'ancien et on réessaie
                        if compact_messages(messages, keep):
                            break
                    else:
                        raise
                    resp = client.chat(messages, tools)
''', "réessai après too_large")
patch(F, '"too_large": FailureKind.QUOTA', '''                kind = {"quota": FailureKind.QUOTA, "timeout": FailureKind.TIMEOUT}.get(e.kind, FailureKind.LLM_ERROR)
                return _fail(res, kind, f"erreur LLM ({e.kind}) : {e}")
''', '''                kind = {"quota": FailureKind.QUOTA, "too_large": FailureKind.QUOTA, "timeout": FailureKind.TIMEOUT}.get(e.kind, FailureKind.LLM_ERROR)
                hint = (" — réduisez le contexte ou augmentez la limite (GROQ_TPM_LIMIT, palier supérieur, modèle à plus grand quota)" if e.kind == "too_large" else "")
                return _fail(res, kind, f"erreur LLM ({e.kind}) : {e}{hint}")
''', "statut d'échec explicite")

# =========================================================================================== agentic_core/toolkit.py
patch("agentic_core/toolkit.py", "MAX_READ_CHARS = 4000", "MAX_READ_CHARS = 6000        # ~1700 tokens : on garde les entrées du LLM petites (quota de tokens/minute)",
      "MAX_READ_CHARS = 4000        # ~1100 tokens : entrées du LLM petites (quota de tokens/minute, palier gratuit = 8000)", "lecture de fichier plafonnée à 4000 car.")

# =========================================================================================== repair_agent/agent.py
patch("repair_agent/agent.py", "analyse du Validator interrompue", "    if vr.status == AdmissionStatus.ACCEPTED:\n        return _escalated(kw, status=\"NOTHING_TO_DO\"",
      '''    if vr.run_status == RunStatus.FAILED_CLOSED:
        un = [UnresolvedIssue(issue_id=i.issue_id, reason="analyse du Validator interrompue : rapport partiel, non fiable pour réparer", needs_human=False) for i in vr.issues]
        return _escalated(kw, vstatus=vstatus, src_hash=src_hash, unresolved=un, summary="le Validator a échoué en cours d'analyse (" +
                          (vr.failure.kind.value if vr.failure else "?") + ") : relancez la validation avant de réparer")
    if vr.status == AdmissionStatus.ACCEPTED:
        return _escalated(kw, status="NOTHING_TO_DO"''', "refus d'un rapport de Validator interrompu")

# =========================================================================================== tests (fichier autonome)
TEST = '''"""Limite de tokens Groq (TPM) : plafond dur, calibrage, 413, compactage du contexte. Autonome (n'utilise aucun helper du projet)."""
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
'''
tp = ROOT / "tests" / "test_tpm_limit.py"
if tp.exists() and tp.read_text(encoding="utf-8") == TEST:
    report.append("  déjà appliqué : tests/test_tpm_limit.py")
else:
    edits[tp] = TEST
    report.append("  créé          : tests/test_tpm_limit.py")

# =========================================================================================== écriture
print("\n".join(report))
if errors:
    print("\nERREUR — rien n'a été modifié :")
    print("\n".join("  - " + e for e in errors))
    sys.exit(1)
changed = [p for p in edits if (not p.exists()) or p.read_text(encoding="utf-8") != edits[p]]
if CHECK:
    print(f"\n--check : {len(changed)} fichier(s) seraient modifiés/créés. Rien n'a été écrit.")
    sys.exit(0)
if not changed:
    print("\nRien à faire : le correctif est déjà entièrement appliqué.")
    sys.exit(0)
bk = ROOT.resolve().parent / f"{ROOT.resolve().name}_backup_tpm.tar.gz"
subprocess.run(["tar", "czf", str(bk), "--exclude=.venv", "--exclude=venv", "--exclude=__pycache__", "--exclude=.pytest_cache", "--exclude=out",
                "-C", str(ROOT.resolve().parent), ROOT.resolve().name], check=True)
for p in changed:
    p.write_text(edits[p], encoding="utf-8")
print(f"\nOK : {len(changed)} fichier(s) modifiés/créés. Sauvegarde : {bk}")
print("Restauration si besoin : tar xzf", bk, "-C ..")
print("Vérifiez : python -m pytest -q --ignore=tests/test_validator.py --ignore=tests/test_handoff.py")