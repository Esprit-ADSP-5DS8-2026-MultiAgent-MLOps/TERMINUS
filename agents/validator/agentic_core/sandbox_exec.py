"""Exécution confinée partagée par Validator (contrôle d'admission) et Repair (tests, métriques avant/après).

Commandes en LISTE BLANCHE : l'agent choisit un type de commande et ses paramètres, jamais une ligne de shell.
  harness      : harnais du profil du contrat (sklearn_tabular, train_only…) — « le harnais est un profil parmi d'autres »
  import_check : importer des modules locaux du projet
  entrypoint   : lancer un script .py du projet déclaré (main.py, train.py…)
  pytest       : tests unitaires du projet (seulement si pytest est installé dans l'environnement d'exécution)
Tout passe par validator.sandbox (Docker si disponible, sinon repli subprocess) sur une COPIE temporaire du projet.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from validator.check_dynamic import HARNESS as SKLEARN_HARNESS
from validator.check_dynamic import _ignore, _snapshot
from validator.entrypoints import ROLES_BY_HARNESS, read_manifest_body, resolve_entrypoints
from validator.fs import walk_project
from validator.sandbox import ensure_venv, pick_sandbox

from .behavior_probe import judge_metric
from .toolkit import j, redact

HARNESSES = {
    "sklearn_tabular": {"script": SKLEARN_HARNESS, "run_main": True,
                        "core_steps": ("import_train", "import_predict", "train", "load_model", "predict_df", "predict_dict")},
    "train_only": {"script": Path(__file__).with_name("harness_train_only.py"), "run_main": False,
                   "core_steps": ("import_train", "train")},
}
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
SCRIPT_RE = re.compile(r"^[\w\-./]+\.py$")
COMMANDS = ("harness", "import_check", "entrypoint", "pytest", "introspect")
INTROSPECT_SCRIPT = Path(__file__).with_name("introspect_script.py")
AUDIT_SCRIPT = Path(__file__).with_name("audit_hook.py")
PROBE_SCRIPT = Path(__file__).with_name("behavior_probe.py")


def derive_requirements(root: Path, cache_dir=None) -> Path | None:
    """requirements.txt absent : dérive les dépendances de pyproject.toml ([project].dependencies) ou d'environment.yml
    (section pip + paquets conda `nom=ver` -> `nom==ver`). Fichier hors du projet, jamais écrit dedans.
    Les paquets non épinglés restent tels quels : c'est ENV.DEPENDENCIES_PINNED qui les signale."""
    lines: list[str] = []
    src = "pyproject.toml"
    py = root / "pyproject.toml"
    if py.is_file():
        try:
            import tomllib
            lines = [str(d) for d in (tomllib.loads(py.read_text(encoding="utf-8")).get("project", {}).get("dependencies") or [])]
        except Exception:
            lines = []
    if not lines:
        for n in ("environment.yml", "environment.yaml"):
            f = root / n
            if not f.is_file():
                continue
            src = n
            try:
                import yaml
                for d in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("dependencies") or []:
                    if isinstance(d, dict):
                        lines += [str(x) for x in d.get("pip") or []]
                    elif isinstance(d, str):
                        d = d.split("::")[-1].strip()
                        if re.split(r"[=<>! ]", d)[0].lower() in ("python", "pip", ""):
                            continue
                        lines.append(re.sub(r"^([A-Za-z0-9_.\-]+)=([^=].*)$", r"\1==\2", d))
            except Exception:
                lines = []
            break
    lines = [x for x in lines if x.strip()]
    if not lines:
        return None
    text = f"# source: {src}\n" + "\n".join(lines) + "\n"
    out = Path(cache_dir or os.environ.get("CV_CACHE_DIR") or Path.home() / ".cache" / "contract-validator") / "derived"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"req_{hashlib.sha256(text.encode()).hexdigest()[:16]}.txt"
    f.write_text(text, encoding="utf-8")
    return f


def num(d):
    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


# Ordre de préférence UNIQUEMENT parmi les clés réellement renvoyées par train() (jamais une métrique inventée).
METRIC_PREFERENCE = ("roc_auc", "auc", "f1", "f1_macro", "f1_weighted", "accuracy", "r2", "rmse", "mae", "mse")


def pick_metric(metrics: dict | None) -> tuple[str | None, str, list[str]]:
    """Métrique principale quand le manifeste n'en déclare pas : lue dans le dict renvoyé par l'entraînement réel."""
    nm = {k: v for k, v in num(metrics).items() if math.isfinite(v)}
    keys = list(nm)
    if not keys:
        return None, "train() ne renvoie aucune métrique numérique", []
    if len(keys) == 1:
        return keys[0], "seule métrique numérique renvoyée par train()", keys
    low = {str(k).lower(): k for k in keys}
    for pref in METRIC_PREFERENCE:
        if pref in low:
            return low[pref], f"plusieurs métriques renvoyées {keys} : préférence par convention", keys
    return keys[0], f"plusieurs métriques renvoyées {keys} : première clé du dict", keys


class SandboxRunner:
    def __init__(self, root, *, sandbox="auto", env="venv", cache_dir=None, timeout=300, memory_mb=4096):
        self.root = Path(root).resolve()
        self.sandbox, self.env, self.cache_dir, self.timeout, self.memory_mb = sandbox, env, cache_dir, timeout, memory_mb
        self.info: dict | None = None
        self._sb = None
        self._cache: dict = {}

    # ------------------------------------------------------------------ préparation (une fois)
    def prepare(self, root: Path | None = None) -> str | None:
        """Retourne None si prêt, sinon un message d'erreur (jamais d'exception : ce sera un NOT_CHECKED).
        Le sandbox est préparé PAR projet (requirements.txt différents entre l'original et la copie corrigée) et mis en cache."""
        root = Path(root or self.root).resolve()
        req = root / "requirements.txt"
        dep = req if req.is_file() else derive_requirements(root, self.cache_dir)
        key = (str(root), dep.read_bytes() if dep else None)
        if key in self._cache:
            self._sb, self.info = self._cache[key]
            return None
        try:
            sb, _note = pick_sandbox(self.sandbox, root, self.memory_mb) if req.is_file() or self.sandbox == "subprocess" \
                else pick_sandbox("subprocess", root, self.memory_mb)
        except Exception as e:
            return f"sandbox indisponible : {type(e).__name__}"
        if sb.name == "subprocess":
            if self.env == "venv":
                if dep is None:
                    return ("requirements.txt absent : impossible de construire l'environnement d'exécution isolé "
                            "(aucune dépendance exploitable dans requirements.txt, pyproject.toml ou environment.yml)")
                _tb = time.time()
                py, err = ensure_venv(dep, self.cache_dir, timeout=3600 if self.memory_mb > 4096 else 900)
                self._build_s = time.time() - _tb
                if err:
                    return "installation des dépendances impossible : " + err[-400:]
                sb.python = str(py)
            elif self.env not in ("current", ""):
                sb.python = self.env                      # chemin d'un interpréteur choisi par l'appelant
        self._sb, self.info = sb, sb.describe()
        self.info["environment"] = self._environment_report(sb, root, dep)
        self._cache[key] = (self._sb, self.info)
        return None

    def _environment_report(self, sb, root: Path, dep: Path | None) -> dict:
        from .env_check import inspect_environment
        mode = "docker" if sb.name == "docker" else "venv" if self.env == "venv" else "current" if self.env in ("current", "") else "custom"
        src = None
        if dep is not None:
            src = "requirements.txt" if dep.parent == root else (dep.read_text(encoding="utf-8").splitlines() or ["# source: ?"])[0].replace("# source:", "").strip()
        return inspect_environment(mode=mode, backend=sb.name, python=getattr(sb, "python", None), dep_file=dep, deps_source=src,
                                   network_isolated=bool(getattr(sb, "network_isolated", False)),
                                   build_s=getattr(self, "_build_s", None) if mode == "venv" else None)

    # ------------------------------------------------------------------ primitives
    def _workdir(self, root: Path | None = None):
        work = Path(tempfile.mkdtemp(prefix="agc_"))
        proj = work / "project"
        shutil.copytree(root or self.root, proj, symlinks=True, ignore=_ignore)
        for d in ("out_main", "out_h1", "out_h2", "out_h3", "tmp", "harness"):
            (work / d).mkdir()
        return work, proj

    def run(self, command: str, root: Path | None = None, **p) -> dict:
        """Point d'entrée unique. Retourne un dict de FAITS ; en cas d'impossibilité : {'error': ..., 'not_run': True}."""
        if command not in COMMANDS:
            return {"error": f"commande non autorisée : {command} (autorisées : {list(COMMANDS)})", "not_run": True}
        err = self.prepare(root)
        if err:
            return {"error": err, "not_run": True}
        work, proj = self._workdir(root)
        try:
            return getattr(self, f"_{command}")(work, proj, **p)
        except TypeError as e:
            return {"error": f"paramètres invalides : {e}", "not_run": True}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _res(r) -> dict:
        return {"returncode": r.returncode, "timed_out": r.timed_out, "duration_s": round(r.duration_s, 1),
                **({"stderr_tail": redact(r.stderr.strip()[-500:])} if r.returncode != 0 else {}),
                "stdout_tail": redact(r.stdout.strip()[-300:])}

    def _import_check(self, work, proj, modules):
        if not isinstance(modules, list) or not modules or not all(isinstance(m, str) and MODULE_RE.match(m) for m in modules):
            return {"error": "modules : liste de noms de modules Python valides", "not_run": True}
        code = "import importlib,sys;sys.path.insert(0,'.');[importlib.import_module(m) for m in %r]" % modules[:20]
        r = self._sb.run(["python", "-c", code], work, proj, {"OUTPUT_DIR": "{W}/out_main"}, min(self.timeout, 120))
        return {"command": "import_check", "modules": modules[:20], **self._res(r), "ok": r.returncode == 0 and not r.timed_out}

    def _entrypoint(self, work, proj, script):
        if not isinstance(script, str) or not SCRIPT_RE.match(script) or ".." in script or not (proj / script).is_file():
            return {"error": "script : chemin relatif d'un .py existant du projet", "not_run": True}
        before = _snapshot(proj)
        r = self._sb.run(["python", script], work, proj, {"OUTPUT_DIR": "{W}/out_main"}, self.timeout)
        changed = sorted(k for k, v in _snapshot(proj).items() if before.get(k) != v)
        return {"command": "entrypoint", "script": script, **self._res(r), "ok": r.returncode == 0 and not r.timed_out,
                "files_written_outside_output_dir": changed[:8]}

    def _pytest(self, work, proj):
        if not (proj / "tests").is_dir() and not list(proj.glob("test_*.py")) and not list(proj.glob("*_test.py")):
            return {"command": "pytest", "skipped": "aucun test unitaire dans le projet", "ok": None}
        probe = self._sb.run(["python", "-c", "import pytest"], work, proj, {}, 60)
        if probe.returncode != 0:
            return {"command": "pytest", "skipped": "pytest non installé dans l'environnement d'exécution du projet", "ok": None}
        r = self._sb.run(["python", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"], work, proj, {"OUTPUT_DIR": "{W}/out_main"}, self.timeout)
        return {"command": "pytest", **self._res(r), "ok": r.returncode == 0 and not r.timed_out}

    def _introspect(self, work, proj, modules, symbols):
        """Introspection des paquets INSTALLÉS dans l'environnement du projet (jamais du code du candidat) : voir introspect_script.py."""
        ok = lambda xs: isinstance(xs, list) and all(isinstance(x, str) and MODULE_RE.match(x) for x in xs)
        if not (ok(modules) and ok(symbols)):
            return {"error": "modules et symboles : noms Python valides", "not_run": True}
        shutil.copy(INTROSPECT_SCRIPT, work / "harness" / "_introspect.py")
        (work / "request.json").write_text(json.dumps({"modules": modules[:200], "symbols": symbols[:400]}))
        r = self._sb.run(["python", "{W}/harness/_introspect.py", "--request", "{W}/request.json", "--report", "{W}/introspect.json"],
                         work, proj, {}, min(self.timeout, 180))
        rp = work / "introspect.json"
        if not rp.exists():
            return {"error": redact((r.stderr or "").strip()[-200:]) or "introspection sans rapport", "not_run": True}
        return {"command": "introspect", **json.loads(rp.read_text())}

    def _harness(self, work, proj, profile, data_path, target, primary_metric, check_determinism=True, entrypoints=None):
        h = HARNESSES.get(profile)
        if h is None:
            return {"error": f"profil de harnais inconnu : {profile}", "not_run": True}
        dp = (proj / data_path).resolve()
        if not str(dp).startswith(str(proj.resolve())) or not dp.is_file():
            return {"error": f"données introuvables : {data_path}", "not_run": True}
        shutil.copy(h["script"], work / "harness" / "_harness.py")
        shutil.copy(AUDIT_SCRIPT, work / "harness" / "_audit.py")        # hook d'audit : installé avant tout code du candidat
        shutil.copy(PROBE_SCRIPT, work / "harness" / "_probe.py")        # sonde comportementale (étiquettes permutées)
        # entrypoints : manifeste d'abord, AST ensuite (validator/entrypoints.py). Résolu PAR PYTHON sur la copie du projet.
        roles = ROLES_BY_HARNESS.get(profile, ("train",))
        files, _links = walk_project(proj)
        eps = resolve_entrypoints(proj, files, read_manifest_body(proj), profile)
        if entrypoints is None and eps.ambiguous(roles):
            return {"error": "entrypoints ambigus : " + "; ".join(f"{p.role} -> {p.candidates[:4]}" for p in eps.ambiguous(roles)),
                    "not_run": True, "entrypoints": eps.summary()}
        ep_map = entrypoints if entrypoints is not None else eps.harness_map(roles)
        (work / "manifest.json").write_text(json.dumps({"data_path": data_path, "target": target, "task": "n/a", "primary_metric": primary_metric,
                                                         "entrypoints": ep_map}))
        before = _snapshot(proj)
        main_file = eps.main if eps.main and (proj / eps.main).is_file() else None
        main = self._sb.run(["python", "{W}/harness/_audit.py", "{W}/audit_main.jsonl", "{W}/project", main_file], work, proj,
                            {"OUTPUT_DIR": "{W}/out_main"}, self.timeout) if h["run_main"] and main_file else None
        args = ["python", "{W}/harness/_harness.py", "--project", "{W}/project", "--manifest", "{W}/manifest.json",
                "--out1", "{W}/out_h1", "--out2", "{W}/out_h2", "--out3", "{W}/out_h3", "--report", "{W}/report.json"]
        if not check_determinism:
            args.append("--no-determinism")
        hr = self._sb.run(args, work, proj, {}, self.timeout)
        rp = work / "report.json"
        rep = json.loads(rp.read_text()) if rp.exists() else {}
        changed = sorted(k for k, v in _snapshot(proj).items() if before.get(k) != v)
        steps = {k: ("ok" if v.get("ok") else f"FAILED: {v.get('error', '')[:200]}") for k, v in (rep.get("steps") or {}).items()}
        m1, m2 = rep.get("metrics"), rep.get("metrics2")
        det, diff = None, {}
        if m1 and m2:
            diff = {k: round(abs(num(m1)[k] - num(m2)[k]), 6) for k in num(m1) if k in num(m2) and abs(num(m1)[k] - num(m2)[k]) > 1e-9}
            det = not diff
        res = {"command": "harness", "profile": profile, "sandbox": {"backend": self._sb.name, "network_isolated": self._sb.network_isolated},
               "entrypoints": {r: f"{e.get('file')}:{e.get('function')}" for r, e in ep_map.items()},
               "contract_steps": steps, "metrics": num(m1) or None, "second_run_metrics": num(m2) or None,
               "deterministic": det, "metric_differences": diff, "model_artifact_in_output_dir": rep.get("artifact_exists"),
               "files_written_outside_output_dir": changed[:8], "peak_mem_mb": rep.get("peak_mem_mb")}
        if main is not None:
            res["main"] = {**self._res(main), "wrote_model_artifact": (work / "out_main" / "model.joblib").exists()}
        if profile == "sklearn_tabular":
            res["train_returned_dict"] = m1 is not None
            res["predict"] = {"rows_out_for_5_rows": rep.get("predict_len"), "rows_out_for_dict": rep.get("predict_dict_len"),
                              "json_serializable": rep.get("predict_json_ok")}
        events = []
        for name in ("audit_main.jsonl", "audit_harness.jsonl"):
            af = work / name
            for ln in (af.read_text(encoding="utf-8", errors="replace").splitlines()[:200] if af.exists() else []):
                try:
                    events.append(json.loads(ln))
                except ValueError:
                    pass
        res["audit"] = {"n_events": len(events), "events": events[:40]}
        if hr.timed_out:
            res["timeout"] = f"harnais > {self.timeout}s"
        if rep.get("fatal"):
            res["harness_error"] = redact(rep["fatal"][-300:])
        if not rp.exists() and not hr.timed_out:
            res["harness_error"] = redact(hr.stderr.strip()[-300:])
        core_ok = all(steps.get(s) == "ok" for s in h["core_steps"])
        if not primary_metric:                       # métrique non déclarée : on la LIT dans ce que train() a réellement renvoyé
            primary_metric, how, cands = pick_metric(m1)
            res["metric_discovery"] = {"chosen": primary_metric, "rule": how, "candidates": cands}
        res["primary_metric"] = primary_metric
        res["perm_metrics"] = num(rep.get("perm_metrics")) or None
        if rep.get("probe_error"):
            res["probe_error"] = redact(str(rep["probe_error"])[:200])
        # verdict MESURÉ de ML.METRICS_GENUINE (voir behavior_probe.judge_metric) : étiquettes permutées + fuite
        res["metric_probe"] = judge_metric(primary_metric, m1, rep.get("perm_metrics"), rep.get("probe"), rep.get("probe_error"))
        metric_ok = isinstance(num(m1).get(primary_metric), (int, float)) and math.isfinite(num(m1)[primary_metric])
        main_ok = main is None or (main.returncode == 0 and not main.timed_out)
        res["executed_ok"] = bool(main_ok and core_ok and metric_ok and not rep.get("fatal"))
        res["_summary"] = "exécution réussie" if res["executed_ok"] else "exécution NON réussie : voir contract_steps / main"
        return res


def metrics_delta(before: dict | None, after: dict | None):
    b, a = num(before), num(after)
    return {k: round(a[k] - b[k], 6) for k in b if k in a}
