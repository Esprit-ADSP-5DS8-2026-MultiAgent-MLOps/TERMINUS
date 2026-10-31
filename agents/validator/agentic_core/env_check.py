"""Vérification de l'ENVIRONNEMENT d'exécution : le projet tourne-t-il dans SON venv isolé, avec SES dépendances ?
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
        if not ok:   # environnement sans pip (créé par uv) : mêmes informations via importlib.metadata
            listing = ("import json,importlib.metadata as m;print(json.dumps([{'name':d.metadata['Name'],"
                       "'version':d.version} for d in m.distributions() if d.metadata['Name']]))")
            ok, out, err = _probe(python, ["-c", listing])
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
