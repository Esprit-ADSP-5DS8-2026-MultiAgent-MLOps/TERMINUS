#!/usr/bin/env python3
"""Validator : (1) trace des ÉTAPES effectuées (rapport JSON + `--steps`), (2) VÉRIFICATION que le projet s'exécute dans
SON environnement isolé, dans le sandbox (contrôle ENV.SANDBOX_ISOLATION, calculé par Python, jamais par le LLM).
Prérequis : apply_dl_support.py déjà appliqué. `validator/` (baseline) n'est pas touché. Idempotent."""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent


def patch(rel, old, new, marker):
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    if marker in s:
        print(f"= {rel} (déjà appliqué : {marker[:30]})")
        return
    if old not in s:
        sys.exit(f"!! {rel} : ancre introuvable ({old[:50]!r}). Appliquez d'abord apply_dl_support.py, sur l'archive d'origine.")
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"+ {rel}")


def write(rel, text):
    (ROOT / rel).write_text(text, encoding="utf-8")
    print(f"+ {rel}")


# ============================================================================ nouveaux modules
write("agentic_core/trace.py", r'''"""Trace des étapes d'un agent : phases (contrôles statiques, environnement, exécution, LLM, politique) + appels d'outils.
Sert au rapport (champ `steps`), à `--steps` et à la progression en direct. Aucune décision n'en dépend."""
from __future__ import annotations

import time
from contextlib import contextmanager


class Trace:
    def __init__(self, progress=None, agent="validator"):
        self.t0 = time.time()
        self.progress, self.agent = progress, agent
        self.phases: list[dict] = []

    def _emit(self, rec):
        fn = getattr(self.progress, "on_step", None)
        if fn:
            try:
                fn(self.agent, rec["name"], rec.get("detail", ""), rec["status"], rec["duration_s"])
            except Exception:
                pass

    @contextmanager
    def phase(self, name: str, detail: str = ""):
        rec = {"kind": "phase", "name": name, "detail": detail, "status": "ok", "ts": time.time(), "duration_s": 0.0}
        try:
            yield rec                                     # le corps peut écrire rec["detail"] / rec["status"]
        except BaseException as e:
            rec["status"] = f"erreur : {type(e).__name__}"
            raise
        finally:
            rec["duration_s"] = round(time.time() - rec["ts"], 2)
            self.phases.append(rec)
            self._emit(rec)

    def event(self, name: str, detail: str = "", status: str = "ok"):
        rec = {"kind": "phase", "name": name, "detail": detail, "status": status, "ts": time.time(), "duration_s": 0.0}
        self.phases.append(rec)
        self._emit(rec)

    def export(self, toolbox=None) -> list[dict]:
        rows = list(self.phases) + list(getattr(toolbox, "trace", []) or [])
        rows.sort(key=lambda r: r.get("ts", 0))
        out = []
        for n, r in enumerate(rows, 1):
            out.append({"n": n, "kind": r["kind"], "name": r["name"], "detail": str(r.get("detail", ""))[:200],
                        "status": r.get("status", "ok"), "t": round(r.get("ts", self.t0) - self.t0, 2),
                        "duration_s": r.get("duration_s", 0.0)})
        return out
''')

write("agentic_core/env_check.py", r'''"""Vérification de l'ENVIRONNEMENT d'exécution : le projet tourne-t-il dans SON venv isolé, avec SES dépendances ?
Faits uniquement (aucune décision) : le contrôle ENV.SANDBOX_ISOLATION les interprète côté Python."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def pinned_requirements(dep_file: Path | None) -> dict[str, str]:
    """{nom canonique: spécificateur} pour les dépendances épinglées par `==`."""
    out: dict[str, str] = {}
    if not dep_file or not Path(dep_file).is_file():
        return out
    for raw in Path(dep_file).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            r = Requirement(line)
        except Exception:
            continue
        ops = [s for s in r.specifier if s.operator == "==" and "*" not in s.version]
        if ops:
            out[str(canonicalize_name(r.name))] = f"=={ops[0].version}"
    return out


def _probe(python: str, args: list[str], timeout: int = 60):
    try:
        r = subprocess.run([python, *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip(), r.stderr.strip()[-200:]
    except Exception as e:
        return False, "", f"{type(e).__name__}: {e}"[:200]


def inspect_environment(*, mode: str, backend: str, python: str | None, dep_file: Path | None, deps_source: str | None,
                        network_isolated: bool, build_s: float | None = None) -> dict:
    env = {"mode": mode, "backend": backend, "python": python, "network_isolated": bool(network_isolated),
           "deps_source": deps_source, "build_s": None if build_s is None else round(build_s, 1),
           "deps_sha256": (hashlib.sha256(Path(dep_file).read_bytes()).hexdigest()[:16] if dep_file and Path(dep_file).is_file() else None),
           "is_virtualenv": None, "shares_validator_env": None, "own_venv": False, "pinned_checked": 0,
           "mismatches": [], "missing": [], "verified": False, "note": ""}
    if backend == "docker":
        env["note"] = "environnement construit dans l'image Docker (non inspecté depuis l'hôte)"
        env["own_venv"] = True
        return env
    if not python:
        env["note"] = "interpréteur inconnu"
        return env
    ok, out, err = _probe(python, ["-c", "import sys,json;print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix,'version':sys.version.split()[0]}))"])
    if not ok:
        env["note"] = f"interpréteur inutilisable : {err}"
        return env
    info = json.loads(out)
    env["python_version"], env["prefix"] = info["version"], info["prefix"]
    env["is_virtualenv"] = info["prefix"] != info["base"]
    env["shares_validator_env"] = Path(info["prefix"]).resolve() == Path(sys.prefix).resolve()
    env["own_venv"] = bool(mode == "venv" and env["is_virtualenv"] and not env["shares_validator_env"])
    pins = pinned_requirements(dep_file)
    if pins:
        ok, out, err = _probe(python, ["-m", "pip", "list", "--format=json", "--disable-pip-version-check"])
        if ok:
            installed = {str(canonicalize_name(p["name"])): p["version"] for p in json.loads(out)}
            for name, spec in pins.items():
                have = installed.get(name)
                if have is None:
                    env["missing"].append(f"{name}{spec}")
                elif not SpecifierSet(spec).contains(have, prereleases=True):
                    env["mismatches"].append({"package": name, "required": spec, "installed": have})
            env["pinned_checked"] = len(pins)
            env["verified"] = True
        else:
            env["note"] = f"pip list impossible : {err}"
    else:
        env["verified"] = env["is_virtualenv"] is True and not env["shares_validator_env"]
        env["note"] = "aucune dépendance épinglée à comparer"
    return env
''')

# ============================================================================ schéma : 2 champs au rapport
patch("agentic_core/schemas.py",
      "    sandbox: Optional[dict[str, Any]] = None\n    summary: Optional[str] = None\n    duration_s: float = 0.0\n\n\n# ----------------------------------------------------------------------------- Repair",
      "    sandbox: Optional[dict[str, Any]] = None\n    environment: Optional[dict[str, Any]] = None      # environnement d'exécution vérifié (venv isolé, dépendances conformes)\n    steps: list[dict[str, Any]] = Field(default_factory=list)   # étapes effectuées, dans l'ordre (phases + appels d'outils)\n    summary: Optional[str] = None\n    duration_s: float = 0.0\n\n\n# ----------------------------------------------------------------------------- Repair",
      "environment: Optional[dict[str, Any]]")

# ============================================================================ trace des appels d'outils (tous agents)
patch("agentic_core/toolkit.py", "import json\nimport re\n", "import json\nimport re\nimport time\n", "import time\n")
patch("agentic_core/toolkit.py", "        self.calls: list[dict] = []\n", "        self.calls: list[dict] = []\n        self.trace: list[dict] = []\n", "self.trace: list[dict]")
patch("agentic_core/toolkit.py",
      "        fn = self.tools.get(name)\n        try:\n            if fn is None:",
      "        fn = self.tools.get(name)\n        _t0 = time.time()\n        try:\n            if fn is None:",
      "_t0 = time.time()")
patch("agentic_core/toolkit.py",
      "        self.calls.append({\"tool\": name, \"args\": j(args or {})[:160]})\n",
      "        self.calls.append({\"tool\": name, \"args\": j(args or {})[:160]})\n        self.trace.append({\"kind\": \"tool\", \"name\": name, \"detail\": j(args or {})[:160], \"ts\": _t0,\n                           \"duration_s\": round(time.time() - _t0, 2), \"status\": \"erreur\" if out.startswith('{\"error\"') else \"ok\"})\n",
      "self.trace.append")

# ============================================================================ sandbox : rapport d'environnement
patch("agentic_core/sandbox_exec.py", "import tempfile\n", "import tempfile\nimport time\n", "import time\n")
patch("agentic_core/sandbox_exec.py", "    lines: list[str] = []\n    py = root / \"pyproject.toml\"",
      "    lines: list[str] = []\n    src = \"pyproject.toml\"\n    py = root / \"pyproject.toml\"", "src = \"pyproject.toml\"")
patch("agentic_core/sandbox_exec.py", "            if not f.is_file():\n                continue\n            try:\n                import yaml",
      "            if not f.is_file():\n                continue\n            src = n\n            try:\n                import yaml", "src = n\n")
patch("agentic_core/sandbox_exec.py", '    text = "\\n".join(lines) + "\\n"\n', '    text = f"# source: {src}\\n" + "\\n".join(lines) + "\\n"\n', "# source: {src}")
patch("agentic_core/sandbox_exec.py",
      "                py, err = ensure_venv(dep, self.cache_dir, timeout=3600 if self.memory_mb > 4096 else 900)",
      "                _tb = time.time()\n                py, err = ensure_venv(dep, self.cache_dir, timeout=3600 if self.memory_mb > 4096 else 900)\n                self._build_s = time.time() - _tb",
      "self._build_s = time.time()")
patch("agentic_core/sandbox_exec.py",
      "        self._sb, self.info = sb, sb.describe()\n        self._cache[key] = (self._sb, self.info)",
      "        self._sb, self.info = sb, sb.describe()\n        self.info[\"environment\"] = self._environment_report(sb, root, dep)\n        self._cache[key] = (self._sb, self.info)",
      "self._environment_report(sb")
patch("agentic_core/sandbox_exec.py", "    # ------------------------------------------------------------------ primitives\n",
      '''    def _environment_report(self, sb, root: Path, dep: Path | None) -> dict:
        from .env_check import inspect_environment
        mode = "docker" if sb.name == "docker" else "venv" if self.env == "venv" else "current" if self.env in ("current", "") else "custom"
        src = None
        if dep is not None:
            src = "requirements.txt" if dep.parent == root else (dep.read_text(encoding="utf-8").splitlines() or ["# source: ?"])[0].replace("# source:", "").strip()
        return inspect_environment(mode=mode, backend=sb.name, python=getattr(sb, "python", None), dep_file=dep, deps_source=src,
                                   network_isolated=bool(getattr(sb, "network_isolated", False)),
                                   build_s=getattr(self, "_build_s", None) if mode == "venv" else None)

    # ------------------------------------------------------------------ primitives
''', "def _environment_report")

# ============================================================================ contrôle ENV.SANDBOX_ISOLATION (Python seul)
patch("validator_agent/checks.py", "def merge_llm_over_python(",
      "PYTHON_ONLY_CHECKS = {\"ENV.SANDBOX_ISOLATION\"}      # jamais décidés ni modifiés par le LLM\n\n\ndef merge_llm_over_python(", "PYTHON_ONLY_CHECKS = ")
patch("validator_agent/checks.py", "    for c in llm_checks:\n        cur = out.get(c.check_id)\n        if cur is None:",
      "    for c in llm_checks:\n        if c.check_id in PYTHON_ONLY_CHECKS:\n            continue\n        cur = out.get(c.check_id)\n        if cur is None:", "if c.check_id in PYTHON_ONLY_CHECKS")
patch("validator_agent/checks.py", "def apply_harness(tb, contract: AdmissionContract, outcome: DeterministicOutcome, *, auto_execute: bool) -> DeterministicOutcome:",
      '''def apply_harness(tb, contract: AdmissionContract, outcome: DeterministicOutcome, *, auto_execute: bool) -> DeterministicOutcome:
    outcome = _apply_harness_core(tb, contract, outcome, auto_execute=auto_execute)
    return apply_environment_check(tb, contract, outcome)


def apply_environment_check(tb, contract: AdmissionContract, outcome: DeterministicOutcome) -> DeterministicOutcome:
    """ENV.SANDBOX_ISOLATION : le projet a-t-il tourné dans SON environnement isolé, avec SES dépendances ? Python seul, sur des faits."""
    s = contract.check("ENV.SANDBOX_ISOLATION")
    if s is None:
        return outcome
    env = (tb.sandbox_info or {}).get("environment")
    if not env:
        chk = _mk(s, CheckStatus.NOT_CHECKED, note="aucune exécution dans le sandbox : environnement non vérifiable", origin="validator", source="python")
        issue = None
    else:
        mode, bad = env.get("mode"), (env.get("mismatches") or []) + [{"missing": m} for m in env.get("missing") or []]
        summary = (f"mode={mode} · backend={env.get('backend')} · python={env.get('python')} · venv={env.get('is_virtualenv')} · "
                   f"dépendances {env.get('deps_source') or 'aucune'} · {env.get('pinned_checked', 0)} épinglée(s) comparée(s) · "
                   f"réseau isolé={env.get('network_isolated')}")
        tb.observe(summary)
        if not env.get("network_isolated") and env.get("backend") != "docker":
            outcome.limits.append("sandbox sans isolation réseau (repli sous-processus) : le garde-fou statique reste la protection principale")
        if mode == "current" or env.get("shares_validator_env"):
            chk = _mk(s, CheckStatus.FAIL, evidence=summary, note="le projet s'exécute dans l'interpréteur du validateur, pas dans son propre environnement", origin="validator", source="python")
            issue = _issue(s, "Exécution hors environnement dédié (--env current) : dépendances du validateur utilisées.", evidence=summary)
        elif bad:
            ev = summary + " · écarts : " + json.dumps(bad[:5], ensure_ascii=False)
            tb.observe(ev)
            chk = _mk(s, CheckStatus.FAIL, evidence=ev, note="l'environnement ne contient pas les versions épinglées du projet", origin="harness", source="python")
            issue = _issue(s, "L'environnement d'exécution ne correspond pas aux dépendances épinglées du projet.", evidence=ev)
        elif env.get("backend") == "docker":
            chk = _mk(s, CheckStatus.PASS, evidence=summary, note="image Docker construite pour le projet (environnement non inspecté depuis l'hôte)", source="python")
            issue = None
        elif env.get("own_venv") or (mode == "custom" and env.get("is_virtualenv") and env.get("verified") and env.get("pinned_checked")):
            note = ("venv dédié construit pour ce projet" if env.get("own_venv") else "interpréteur fourni (--env) conforme aux dépendances épinglées du projet")
            chk = _mk(s, CheckStatus.PASS, evidence=summary, note=note, source="python")
            issue = None
        else:
            chk = _mk(s, CheckStatus.NOT_CHECKED, evidence=summary, note=env.get("note") or "environnement non vérifiable", origin="validator", source="python")
            issue = None
    outcome.checks = [c for c in outcome.checks if c.check_id != s.id] + [chk]
    if issue and chk.status == CheckStatus.FAIL:
        outcome.issues = [i for i in outcome.issues if i.check_id != s.id] + [issue]
    return outcome


def _apply_harness_core(tb, contract: AdmissionContract, outcome: DeterministicOutcome, *, auto_execute: bool) -> DeterministicOutcome:''',
      "def apply_environment_check")

# ============================================================================ agent : phases + champs du rapport
patch("validator_agent/agent.py", "    phash, nfiles = compute_project_hash(root)\n",
      "    trace = Trace(progress, \"validator\")\n    phash, nfiles = compute_project_hash(root)\n", "trace = Trace(progress")
patch("validator_agent/agent.py", "        py_out = run_static_checks(tb, contract)\n        py_out = apply_harness(tb, contract, py_out, auto_execute=cfg.auto_execute)\n",
      '''        with trace.phase("contrôles statiques", "AST, dépendances, secrets, réseau, données"):
            py_out = run_static_checks(tb, contract)
        with trace.phase("environnement + exécution sandbox", f"env={cfg.env} · sandbox={cfg.sandbox}") as _ph:
            py_out = apply_harness(tb, contract, py_out, auto_execute=cfg.auto_execute)
            _e = (tb.sandbox_info or {}).get("environment") or {}
            _ph["detail"] = (f"mode={_e.get('mode', '—')} · backend={_e.get('backend', '—')} · venv dédié={_e.get('own_venv', '—')} · "
                             f"exécution {'réussie' if tb.executed_ok else 'non réussie' if tb.executed_ok is False else 'non lancée'}")
            _ph["status"] = "ok" if tb.executed_ok else "attention"
''', "with trace.phase(\"contrôles statiques\"")
patch("validator_agent/agent.py", "            tb.sandbox_info = hit.get(\"sandbox_info\")\n            cached_deterministic = True\n",
      "            tb.sandbox_info = hit.get(\"sandbox_info\")\n            cached_deterministic = True\n            trace.event(\"cache déterministe\", \"contrôles Python et exécution rechargés (projet inchangé)\")\n", "trace.event(\"cache déterministe\"")
patch("validator_agent/agent.py", "        res = run_loop(client, spec, loop_cfg, progress)\n",
      "        with trace.phase(\"interprétation LLM\", f\"{len(open_ids) if open_ids is not None else 'tous'} contrôle(s) à interpréter\"):\n            res = run_loop(client, spec, loop_cfg, progress)\n", "trace.phase(\"interprétation LLM\"")
patch("validator_agent/agent.py", "    else:\n        res.stats = dict(EMPTY_STATS)\n",
      "    else:\n        res.stats = dict(EMPTY_STATS)\n        trace.event(\"interprétation LLM\", \"non nécessaire (tout est décidé par Python)\", \"ignorée\")\n", "non nécessaire (tout est décidé")
patch("validator_agent/agent.py", "    policy, checks = evaluate(contract, checks, issues, executed_ok=executed_ok)\n",
      "    with trace.phase(\"moteur de politique\", f\"profil {contract.id}\"):\n        policy, checks = evaluate(contract, checks, issues, executed_ok=executed_ok)\n", "trace.phase(\"moteur de politique\"")
patch("validator_agent/agent.py", "        observed_metrics=(tb.harness_result or {}).get(\"metrics\"), sandbox=tb.sandbox_info,\n",
      "        observed_metrics=(tb.harness_result or {}).get(\"metrics\"), sandbox=tb.sandbox_info,\n        environment=(tb.sandbox_info or {}).get(\"environment\"), steps=trace.export(tb),\n", "steps=trace.export(tb)")
patch("validator_agent/agent.py", "from agentic_core.schemas import (AdmissionStatus,", "from agentic_core.trace import Trace\nfrom agentic_core.schemas import (AdmissionStatus,", "from agentic_core.trace import Trace")

# ============================================================================ progression en direct
patch("agentic_core/progress.py", "    def on_end(self, agent, stats, failure=None): ...\n",
      "    def on_end(self, agent, stats, failure=None): ...\n    def on_step(self, agent, name, detail=\"\", status=\"ok\", duration_s=0.0): ...\n", "def on_step(self, agent, name, detail=\"\", status=\"ok\", duration_s=0.0): ...")
patch("agentic_core/progress.py", "    def on_end(self, agent, stats, failure=None):\n        t = self.totals.get(agent, {})",
      "    def on_step(self, agent, name, detail=\"\", status=\"ok\", duration_s=0.0):\n        mark = \"✓\" if status == \"ok\" else \"•\" if status == \"ignorée\" else \"!\"\n        self._p(agent, f\"{mark} {name} · {duration_s:.1f}s\" + (f\" · {detail}\" if detail else \"\") + (\"\" if status in (\"ok\", \"ignorée\") else f\" [{status}]\"))\n\n    def on_end(self, agent, stats, failure=None):\n        t = self.totals.get(agent, {})",
      "mark = \"✓\"")

# ============================================================================ CLI : --steps + résumé d'environnement
patch("validator_agent/__main__.py", "    ap.add_argument(\"--explore\"",
      "    ap.add_argument(\"--steps\", action=\"store_true\", help=\"affiche le détail des étapes effectuées (phases + appels d'outils)\")\n    ap.add_argument(\"--explore\"", "\"--steps\"")
patch("validator_agent/__main__.py", "    for l in rep.limits:\n        print(f\"  limite : {l}\")\n",
      '''    for l in rep.limits:
        print(f"  limite : {l}")
    e = rep.environment
    if e:
        ok = next((c for c in rep.checks if c.check_id == "ENV.SANDBOX_ISOLATION"), None)
        print(f"\\nEnvironnement d'exécution : {ok.status.value if ok else '—'} · mode {e.get('mode')} · {e.get('backend')} · python {e.get('python_version', '?')} "
              f"· venv dédié : {'oui' if e.get('own_venv') else 'non'} · dépendances : {e.get('deps_source') or 'aucune'} "
              f"({e.get('pinned_checked', 0)} épinglée(s) vérifiée(s), {len(e.get('mismatches') or []) + len(e.get('missing') or [])} écart(s)) "
              f"· réseau isolé : {'oui' if e.get('network_isolated') else 'non'}")
    if a.steps:
        print("\\nÉtapes effectuées :")
        for s in rep.steps:
            print(f"  {s['n']:>2}. t+{s['t']:>6.1f}s {s['duration_s']:>6.1f}s  {'PHASE' if s['kind'] == 'phase' else 'outil':<5}  {s['name']}"
                  + (f" — {s['detail']}" if s['detail'] else "") + ("" if s['status'] == 'ok' else f"  [{s['status']}]"))
''', "Étapes effectuées")

# ============================================================================ contrats : le nouveau contrôle
CHK = '''  - {{id: ENV.SANDBOX_ISOLATION, level: security_exec, title: Environnement isolé dans le sandbox, category: execution, nature: methodological, requirement: {req}, on_fail: NEEDS_REVIEW,
     description: "Le projet s'exécute dans SON venv dédié (ou un interpréteur fourni conforme à ses dépendances épinglées), jamais dans l'interpréteur du validateur. Calculé par Python (rapport `environment`), jamais par le LLM."}}
'''
for name, req in (("sklearn_tabular", "mandatory"), ("training_only", "mandatory"), ("deep_learning", "mandatory")):
    p = ROOT / f"policy/contracts/{name}.yaml"
    s = p.read_text(encoding="utf-8")
    if "ENV.SANDBOX_ISOLATION" in s:
        print(f"= policy/contracts/{name}.yaml (déjà appliqué)")
        continue
    p.write_text(s.rstrip("\n") + "\n" + CHK.format(req=req), encoding="utf-8")
    print(f"+ policy/contracts/{name}.yaml ({req})")
print("OK — régénérez les schémas : python -m agentic_core.export_schemas docs/schemas")

# ============================================================================ tests : faux sandbox + nouveaux tests
patch("tests/helpers.py",
      '    info = {"backend": "fake", "network_isolated": True}\n',
      '    info = {"backend": "fake", "network_isolated": True,\n'
      '            "environment": {"mode": "venv", "backend": "fake", "python": "/fake/venv/bin/python", "is_virtualenv": True,\n'
      '                            "shares_validator_env": False, "own_venv": True, "deps_source": "requirements.txt", "pinned_checked": 4,\n'
      '                            "mismatches": [], "missing": [], "network_isolated": True, "verified": True}}\n',
      '"environment": {"mode": "venv"')

TEST = r'''"""Étapes du Validator et vérification de l'environnement isolé (ENV.SANDBOX_ISOLATION)."""
import sys

from agentic_core.env_check import inspect_environment, pinned_requirements
from agentic_core.llm_client import ScriptedClient
from agentic_core.sandbox_exec import derive_requirements
from agentic_core.schemas import AdmissionStatus as S, Check, CheckStatus as C
from helpers import CREDIT, FakeRunner
from policy import load_contract
from validator_agent.agent import ValidatorConfig, run_validator
from validator_agent.checks import merge_llm_over_python

HARNESS = {"executed_ok": True, "metrics": {"roc_auc": 0.9}, "_summary": "exécution réussie",
           "contract_steps": {"import_train": "ok", "import_predict": "ok", "train": "ok", "load_model": "ok",
                              "predict_df": "ok", "predict_dict": "ok"},
           "deterministic": True, "files_written_outside_output_dir": []}
CFG = ValidatorConfig(deterministic=True, auto_execute=True, max_llm_steps=0)


def _run(env=None, info=None):
    r = FakeRunner(harness=HARNESS)
    if env is not None:
        r.info = {**FakeRunner.info, "environment": {**FakeRunner.info["environment"], **env}}
    if info is not None:
        r.info = info
    return run_validator(CREDIT, ScriptedClient([]), "sklearn-tabular", CFG, runner=r)


def _chk(rep):
    return next(c for c in rep.checks if c.check_id == "ENV.SANDBOX_ISOLATION")


def test_report_lists_steps_in_order():
    rep = _run()
    names = [s["name"] for s in rep.steps]
    assert [s["n"] for s in rep.steps] == list(range(1, len(rep.steps) + 1))
    for expected in ("contrôles statiques", "environnement + exécution sandbox", "moteur de politique"):
        assert expected in names
    assert any(s["kind"] == "tool" and s["name"] == "run_in_sandbox" for s in rep.steps)
    assert names.index("contrôles statiques") < names.index("environnement + exécution sandbox") < names.index("moteur de politique")


def test_dedicated_venv_passes():
    rep = _run()
    assert _chk(rep).status == C.PASS and rep.status == S.ACCEPTED
    assert rep.environment["own_venv"] is True


def test_validator_own_interpreter_is_refused():
    rep = _run(env={"mode": "current", "own_venv": False, "shares_validator_env": True})
    assert _chk(rep).status == C.FAIL and rep.status == S.NEEDS_REVIEW
    assert any(i.check_id == "ENV.SANDBOX_ISOLATION" for i in rep.issues)


def test_pinned_version_mismatch_is_refused():
    rep = _run(env={"mismatches": [{"package": "scikit-learn", "required": "==1.8.0", "installed": "1.9.1"}]})
    assert _chk(rep).status == C.FAIL and rep.status == S.NEEDS_REVIEW


def test_no_environment_information_is_not_checked():
    rep = _run(info={"backend": "fake", "network_isolated": True})
    assert _chk(rep).status == C.NOT_CHECKED and rep.status == S.NEEDS_REVIEW


def test_llm_cannot_decide_isolation():
    py = [Check(check_id="ENV.SANDBOX_ISOLATION", status=C.NOT_CHECKED, source="python")]
    llm = [Check(check_id="ENV.SANDBOX_ISOLATION", status=C.PASS, source="llm")]
    assert merge_llm_over_python(py, llm)[0].status == C.NOT_CHECKED
    assert merge_llm_over_python([], llm) == []


def test_inspect_environment_detects_validator_interpreter(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("packaging==0.0.1\n")
    env = inspect_environment(mode="current", backend="subprocess", python=sys.executable, dep_file=req,
                              deps_source="requirements.txt", network_isolated=False)
    assert env["shares_validator_env"] is True and env["own_venv"] is False
    assert env["pinned_checked"] == 1 and env["mismatches"]
    assert pinned_requirements(req) == {"packaging": "==0.0.1"}


def test_dependencies_derived_from_pyproject_and_environment_yml(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "pyproject.toml").write_text('[project]\nname="x"\ndependencies=["scikit-learn==1.5.0","torch==2.4.0"]\n')
    (b / "environment.yml").write_text("dependencies:\n  - python=3.12\n  - numpy=1.26.4\n  - pip:\n    - torch==2.4.0\n")
    ta, tb_ = derive_requirements(a, tmp_path).read_text(), derive_requirements(b, tmp_path).read_text()
    assert "# source: pyproject.toml" in ta and "scikit-learn==1.5.0" in ta
    assert "# source: environment.yml" in tb_ and "numpy==1.26.4" in tb_ and "torch==2.4.0" in tb_
    assert "python" not in tb_.split("\n", 1)[1]
    assert derive_requirements(tmp_path / "nothing", tmp_path) is None


def test_deep_learning_profile():
    c = load_contract("deep-learning")
    assert {"torch", "tensorflow", "keras"} <= set(c.frameworks_in_scope) and "torch" not in c.frameworks_out_of_scope
    assert c.sandbox_memory_mb == 16384 and c.harness == "train_only"
    for name in ("sklearn-tabular", "training-only", "deep-learning"):
        assert "ENV.SANDBOX_ISOLATION" in load_contract(name).mandatory_ids
'''
if not (ROOT / "tests/test_steps_env.py").exists():
    write("tests/test_steps_env.py", TEST)
else:
    print("= tests/test_steps_env.py (déjà présent)")
print("OK — vérifiez : python -m pytest -q tests/test_steps_env.py")
